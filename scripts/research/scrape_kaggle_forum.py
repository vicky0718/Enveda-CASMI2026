"""Scrape the competition's discussion forum and public leaderboard (read-only).

Saves raw JSON under data/external/kaggle_forum/ (git-ignored: these are other
people's posts) and a flat comments table for analysis:

    topics.json            every topic with full comment trees
    leaderboard.json       public leaderboard + team members
    posts.parquet          one row per topic opener / comment / reply, with
                           author, votes, LB rank of the author's team (if any)
    forum_dump.txt         every topic as readable text (newest activity first)
    new_posts.md           digest of posts not seen in the previous scrape (daily refresh)
    kernels_top.json       public notebooks of the competition (titles / votes / authors only)

    python scripts/research/scrape_kaggle_forum.py [--reuse]
"""

import base64
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

COMP_ID = 143318
FORUM_ID = 11293906
API = "https://www.kaggle.com/api/i/"
OUT = Path(__file__).resolve().parents[2] / "data" / "external" / "kaggle_forum"
OUT.mkdir(parents=True, exist_ok=True)


def _headers():
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
        user, key = creds["username"], creds["key"]
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode(),
            "Content-Type": "application/json"}


H = _headers()
CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None)


def call(service_method: str, body: dict, retries: int = 4):
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(API + service_method, data=json.dumps(body).encode(), headers=H)
            with urllib.request.urlopen(req, context=CTX, timeout=60) as r:
                time.sleep(0.4)  # be polite
                return json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries:
                raise
            time.sleep(2 ** (attempt + 1))
            print("  retry", service_method, e)


def _votes(msg) -> int:
    return int((msg.get("votes") or {}).get("totalVotes", 0))


def flatten_comments(comments, topic, depth=1, parent=None):
    for c in comments or []:
        yield {
            "topic_id": topic["id"], "topic_title": topic["title"], "kind": "comment" if depth == 1 else "reply",
            "depth": depth, "id": c.get("id"), "parent_id": parent, "post_date": c.get("postDate"),
            "author_username": (c.get("author") or {}).get("url", "").strip("/"),
            "author_name": (c.get("author") or {}).get("displayName"),
            "author_type": c.get("authorType"), "votes": _votes(c),
            "markdown": c.get("rawMarkdown") or "",
        }
        yield from flatten_comments(c.get("replies"), topic, depth + 1, c.get("id"))


def main(reuse: bool = False):
    if reuse and (OUT / "topics.json").exists() and (OUT / "leaderboard.json").exists():
        return build()
    # ---- leaderboard ----
    lb = call("competitions.LeaderboardService/GetLeaderboard", {"competitionId": COMP_ID})
    (OUT / "leaderboard.json").write_text(json.dumps(lb))
    teams = {t["teamId"]: t for t in lb["teams"]}
    user_rank = {}
    for row in lb["publicLeaderboard"]:
        for mbr in teams.get(row["teamId"], {}).get("teamMembers", []):
            user_rank.setdefault(mbr["userName"], (row["rank"], row["displayScore"], teams[row["teamId"]]["teamName"]))
    print(f"leaderboard: {len(lb['publicLeaderboard'])} teams, {len(user_rank)} users")

    # ---- topics ----
    first = call("discussions.DiscussionsService/GetTopicListByForumId", {"forumId": FORUM_ID, "page": 1})
    n = first["count"]
    listing = first["topics"]
    page = 2
    while len(listing) < n:
        nxt = call("discussions.DiscussionsService/GetTopicListByForumId", {"forumId": FORUM_ID, "page": page})["topics"]
        if not nxt:
            break
        listing += nxt
        page += 1
    print(f"topics: {len(listing)} of {n}")

    full = []
    for i, t in enumerate(listing):
        d = call("discussions.DiscussionsService/GetForumTopicById", {"forumTopicId": t["id"], "includeComments": True})
        ft = d["forumTopic"]
        ft["_listing"] = t
        full.append(ft)
        if (i + 1) % 20 == 0:
            print(f"  fetched {i + 1}/{len(listing)}")
    (OUT / "topics.json").write_text(json.dumps(full))
    return build()


def _fmt(r):
    head = f"## by {r.author_name} ({r.author_type}) lb_rank={r.lb_rank} {str(r.post_date)[:10]}" if r.kind == "topic" \
        else f"{'    ' * r.depth}-- [{r.kind}] {r.author_name} ({r.author_type}) lb_rank={r.lb_rank} votes={r.votes}:"
    pad = "" if r.kind == "topic" else "    " * r.depth
    return head + "\n" + "\n".join(pad + line for line in (r.markdown or "").strip().splitlines())


def dump(posts: pd.DataFrame, seen: set):
    """forum_dump.txt (all topics, newest activity first) and new_posts.md (posts not seen before)."""
    last = posts.groupby("topic_id").post_date.max().sort_values(ascending=False)
    out = []
    for tid in last.index:
        g = posts[posts.topic_id == tid]
        t = g[g.kind == "topic"].iloc[0] if (g.kind == "topic").any() else g.iloc[0]
        out.append(f"######## TOPIC {tid} | votes={t.topic_votes} comments={t.topic_comments} | {t.topic_title}")
        out += [_fmt(r) for r in g.itertuples()]
        out.append("")
    (OUT / "forum_dump.txt").write_text("\n".join(out))
    new = posts[~posts.id.isin(seen)] if seen else posts.iloc[0:0]
    lines = [f"# New forum posts since the previous scrape ({len(new)})", ""]
    for tid, g in new.groupby("topic_id", sort=False):
        lines.append(f"## {g.topic_title.iloc[0]} (topic {tid})")
        lines += [_fmt(r) for r in g.itertuples()] + [""]
    (OUT / "new_posts.md").write_text("\n".join(lines))
    print(f"new posts since previous scrape: {len(new)} in {new.topic_id.nunique()} topics")


def kernels():
    """Public notebooks of the competition, most voted first (metadata only)."""
    import subprocess
    r = subprocess.run(["kaggle", "kernels", "list", "--competition", "enveda-CASMI26-molecule-id-mass-spectra",
                        "--sort-by", "voteCount", "--page-size", "100", "--csv"], capture_output=True, text=True,
                       timeout=300)
    if r.returncode == 0 and r.stdout.strip():
        import io
        k = pd.read_csv(io.StringIO(r.stdout))
        (OUT / "kernels_top.json").write_text(k.to_json(orient="records"))
        print(f"public notebooks: {len(k)}")


def build():
    seen = set(pd.read_parquet(OUT / "posts.parquet").id.dropna()) if (OUT / "posts.parquet").exists() else set()
    lb = json.loads((OUT / "leaderboard.json").read_text())
    full = json.loads((OUT / "topics.json").read_text())
    teams = {t["teamId"]: t for t in lb["teams"]}
    user_rank = {}
    for row in lb["publicLeaderboard"]:
        for mbr in teams.get(row["teamId"], {}).get("teamMembers", []):
            user_rank.setdefault(mbr["userName"], (row["rank"], row["displayScore"], teams[row["teamId"]]["teamName"]))
    rows = []
    for ft in full:
        t = ft["_listing"]
        first_msg = ft.get("firstMessage") or {}
        rows.append({
            "topic_id": t["id"], "topic_title": t["title"], "kind": "topic", "depth": 0, "id": t.get("firstForumMessageId"),
            "parent_id": None, "post_date": t.get("postDate"),
            "author_username": ft.get("authorUserName") or (t.get("authorUser") or {}).get("url", "").strip("/"),
            "author_name": ft.get("authorUserDisplayName"), "author_type": t.get("authorType"),
            "votes": t.get("votes", 0), "markdown": first_msg.get("rawMarkdown") or "",
            "topic_votes": t.get("votes", 0), "topic_comments": t.get("commentCount", 0),
        })
        for r in flatten_comments(ft.get("comments"), {"id": t["id"], "title": t["title"]}):
            r.update(topic_votes=t.get("votes", 0), topic_comments=t.get("commentCount", 0))
            rows.append(r)
    posts = pd.DataFrame(rows)
    posts["lb_rank"] = posts.author_username.map(lambda u: user_rank.get(u, (None,))[0])
    posts["lb_score"] = posts.author_username.map(lambda u: user_rank.get(u, (None, None))[1])
    posts["lb_team"] = posts.author_username.map(lambda u: user_rank.get(u, (None, None, None))[2])
    dump(posts, seen)
    posts.to_parquet(OUT / "posts.parquet")
    print(f"posts: {len(posts)} rows; by top-100 users: {(posts.lb_rank <= 100).sum()}")
    try:
        kernels()
    except Exception as e:  # the notebook list is optional
        print("kernels list failed:", type(e).__name__, e)


if __name__ == "__main__":
    import sys

    main(reuse="--reuse" in sys.argv)
