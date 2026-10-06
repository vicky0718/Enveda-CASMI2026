"""Scrape the competition's discussion forum and public leaderboard (read-only).

Saves raw JSON under data/external/kaggle_forum/ (git-ignored: these are other
people's posts) and a flat comments table for analysis:

    topics.json            every topic with full comment trees
    leaderboard.json       public leaderboard + team members
    posts.parquet          one row per topic opener / comment / reply, with
                           author, votes, LB rank of the author's team (if any)

    python scripts/research/scrape_kaggle_forum.py
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


def build():
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
    posts.to_parquet(OUT / "posts.parquet")
    print(f"posts: {len(posts)} rows; by top-100 users: {(posts.lb_rank <= 100).sum()}")


if __name__ == "__main__":
    import sys

    main(reuse="--reuse" in sys.argv)
