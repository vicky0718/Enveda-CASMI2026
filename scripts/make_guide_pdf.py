"""Build the plain-language PDF guide: the competition goal + what the EDA found.

    PYTHONPATH=src python scripts/make_guide_pdf.py
    -> reports/CASMI2026_Competition_and_EDA_Guide.pdf

Uses the figures and numbers produced by scripts/eda/ (aggregates only).
"""

import os
from datetime import date

import matplotlib
from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from casmi.paths import FIG, ROOT

OUT = ROOT / "reports" / "CASMI2026_Competition_and_EDA_Guide.pdf"

# ---------- fonts & palette ----------
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", os.path.join(FONT_DIR, "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#898781")
BLUE = colors.HexColor("#2a78d6")
BLUE_D = colors.HexColor("#184f95")
GREEN = colors.HexColor("#1baf7a")
ORANGE = colors.HexColor("#eb6834")
WASH_BLUE = colors.HexColor("#eef4fc")
WASH_GREEN = colors.HexColor("#ecf8f3")
WASH_ORANGE = colors.HexColor("#fdf0ea")
WASH_GRAY = colors.HexColor("#f4f3f0")
RULE = colors.HexColor("#e1e0d9")

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

# ---------- styles ----------
S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=26, leading=32, textColor=INK, spaceAfter=10),
    "subtitle": ParagraphStyle("subtitle", fontName="DV", fontSize=13, leading=18, textColor=INK2, spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=17, leading=22, textColor=BLUE_D, spaceBefore=6, spaceAfter=8,
                         keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=12.5, leading=17, textColor=INK, spaceBefore=10, spaceAfter=4,
                         keepWithNext=1),
    "body": ParagraphStyle("body", fontName="DV", fontSize=10, leading=14.5, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=10, leading=14.5, textColor=INK, leftIndent=14,
                             bulletIndent=3, spaceAfter=3),
    "caption": ParagraphStyle("caption", fontName="DV-I", fontSize=8.5, leading=12, textColor=INK2, spaceBefore=3,
                              spaceAfter=10),
    "box": ParagraphStyle("box", fontName="DV", fontSize=9.6, leading=13.8, textColor=INK, spaceAfter=3),
    "boxhead": ParagraphStyle("boxhead", fontName="DV-B", fontSize=10, leading=14, textColor=INK, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.8, leading=12, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=8.8, leading=12, textColor=INK),
    "small": ParagraphStyle("small", fontName="DV", fontSize=8.5, leading=12, textColor=INK2),
    "center": ParagraphStyle("center", fontName="DV", fontSize=10, leading=14, textColor=INK2, alignment=TA_CENTER),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="bullet"):
    return [Paragraph(t, S[style], bulletText="•") for t in items]


def figure(name, caption, width=CONTENT_W):
    path = FIG / f"{name}.png"
    w, h = PILImage.open(path).size
    img = Image(str(path), width=width, height=width * h / w)
    return KeepTogether([img, P(caption, "caption")])


def callout(head, body_paras, wash=WASH_BLUE, bar=BLUE):
    inner = [P(head, "boxhead")] + [p if not isinstance(p, str) else P(p, "box") for p in body_paras]
    t = Table([[inner]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), wash),
        ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return KeepTogether([t, Spacer(1, 8)])


def finding(num, title, what, why, wash=WASH_GREEN, bar=GREEN):
    t = Table([[[P(f"Finding {num}: {title}", "boxhead"),
                 P("<b>What we saw.</b> " + what, "box"),
                 P("<b>Why it matters.</b> " + why, "box")]]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), wash),
        ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def table(rows, col_widths, header=True, align_right_from=1):
    data = [[Paragraph(str(c), S["cellb" if (header and i == 0) else "cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2) if header else ("LINEBELOW", (0, 0), (-1, 0), 0, colors.white),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), WASH_GRAY))
    t.setStyle(TableStyle(style))
    return t


# ---------- illustrations ----------
def _arrow(d, x1, y, x2, color=MUTED):
    d.add(Line(x1, y, x2 - 5, y, strokeColor=color, strokeWidth=1.4))
    d.add(Polygon([x2, y, x2 - 6, y + 3.5, x2 - 6, y - 3.5], fillColor=color, strokeColor=color))


def pipeline_drawing():
    """Molecule -> charged -> weighed -> smashed -> pieces weighed -> spectrum."""
    W, H = CONTENT_W, 2.4 * cm
    d = Drawing(W, H)
    steps = [
        ("1. Molecule", "from a plant,", "microbe or animal"),
        ("2. Give it a", "charge (ionise)", "= the 'adduct'"),
        ("3. Weigh the", "whole ion", "= precursor m/z"),
        ("4. Smash it", "with gas at a set", "collision energy"),
        ("5. Weigh the", "pieces", "= fragment peaks"),
    ]
    n = len(steps)
    gap = 12
    bw = (W - gap * (n - 1)) / n
    y0 = 0.1 * cm
    for i, (a, b, c) in enumerate(steps):
        x = i * (bw + gap)
        fill = WASH_BLUE if i < 4 else WASH_GREEN
        d.add(Rect(x, y0, bw, 2.2 * cm, rx=6, ry=6, fillColor=fill, strokeColor=RULE, strokeWidth=0.8))
        d.add(String(x + bw / 2, y0 + 1.55 * cm, a, fontName="DV-B", fontSize=8.6, fillColor=INK, textAnchor="middle"))
        d.add(String(x + bw / 2, y0 + 1.05 * cm, b, fontName="DV", fontSize=8.2, fillColor=INK, textAnchor="middle"))
        d.add(String(x + bw / 2, y0 + 0.5 * cm, c, fontName="DV-I", fontSize=7.6, fillColor=INK2, textAnchor="middle"))
        if i < n - 1:
            _arrow(d, x + bw + 1, y0 + 1.1 * cm, x + bw + gap - 1)
    return d


def toy_spectrum():
    W, H = CONTENT_W, 5.2 * cm
    d = Drawing(W, H)
    x0, y0, pw, ph = 1.2 * cm, 1.0 * cm, W - 2.0 * cm, H - 1.7 * cm
    d.add(Line(x0, y0, x0 + pw, y0, strokeColor=INK2, strokeWidth=0.8))
    d.add(Line(x0, y0, x0, y0 + ph, strokeColor=INK2, strokeWidth=0.8))
    peaks = [(69, .12), (91, .35), (105, .22), (119, .08), (137, .55), (153, 1.0), (165, .18), (199, .30),
             (227, .42), (255, .65)]
    lo, hi = 50, 270
    for mz, it in peaks:
        x = x0 + (mz - lo) / (hi - lo) * pw
        col = GREEN if mz == 255 else (ORANGE if mz == 153 else BLUE)
        d.add(Rect(x - 2, y0, 4, it * ph * 0.92, fillColor=col, strokeColor=None))
    for mz in (50, 100, 150, 200, 250):
        x = x0 + (mz - lo) / (hi - lo) * pw
        d.add(String(x, y0 - 11, str(mz), fontName="DV", fontSize=7.5, fillColor=MUTED, textAnchor="middle"))
    d.add(String(x0 + pw / 2, 0.05 * cm, "weight of the piece (m/z)", fontName="DV", fontSize=8, fillColor=INK2,
                 textAnchor="middle"))
    d.add(String(x0, y0 + ph + 4, "how much", fontName="DV", fontSize=8, fillColor=INK2))
    xb = x0 + (153 - lo) / (hi - lo) * pw
    d.add(String(xb + 6, y0 + ph * 0.92 - 4, "base peak (tallest = 1.0)", fontName="DV", fontSize=7.5, fillColor=ORANGE))
    xp = x0 + (255 - lo) / (hi - lo) * pw
    d.add(String(xp - 6, y0 + 0.65 * ph * 0.92 + 4, "precursor (whole ion)", fontName="DV", fontSize=7.5,
                 fillColor=GREEN, textAnchor="end"))
    return d


def scoring_drawing():
    W, H = CONTENT_W, 3.6 * cm
    d = Drawing(W, H)
    ranks = [1, 2, 3, 4, 5, 10, 25, "not in list"]
    pts = [1.0, 0.5, 0.333, 0.25, 0.2, 0.1, 0.04, 0.0]
    n = len(ranks)
    bw = (W - 1.2 * cm) / n
    base = 1.0 * cm
    for i, (r, p) in enumerate(zip(ranks, pts)):
        x = 1.2 * cm + i * bw
        h = p * 2.0 * cm
        d.add(Rect(x + 6, base, bw - 12, max(h, 0.5), fillColor=BLUE if p > 0 else RULE, strokeColor=None))
        d.add(String(x + bw / 2, base + h + 4, f"{p:g}" if p else "0", fontName="DV-B", fontSize=8, fillColor=INK,
                     textAnchor="middle"))
        d.add(String(x + bw / 2, base - 12, f"rank {r}" if isinstance(r, int) else r, fontName="DV", fontSize=7.8,
                     fillColor=INK2, textAnchor="middle"))
    d.add(String(0, base + 1.0 * cm, "points", fontName="DV", fontSize=8, fillColor=INK2))
    return d


def on_page(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont("DV", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 1.1 * cm, "Enveda CASMI 2026: competition guide & data findings")
        canvas.drawRightString(PAGE_W - MARGIN, 1.1 * cm, f"page {doc.page}")
        canvas.setStrokeColor(RULE)
        canvas.line(MARGIN, 1.45 * cm, PAGE_W - MARGIN, 1.45 * cm)
    canvas.restoreState()


# ---------- content ----------
def build():
    st = []

    # ===== Cover =====
    st += [Spacer(1, 3.2 * cm),
           P("Enveda CASMI 2026", "title"),
           P("Identifying molecules from their mass spectra", "subtitle"),
           P("A plain-language guide to the Kaggle competition and to what we learned from exploring its data",
             "subtitle"),
           Spacer(1, 0.6 * cm)]
    cover = table([
        ["Competition", "Enveda CASMI 2026: Molecule ID From Mass Spectra (Kaggle)"],
        ["Host / prize", "Enveda Therapeutics · $50,000 total (1st place $16,000)"],
        ["Key dates", "Entry and team-merge deadline 7 Dec 2026 · final submission 14 Dec 2026"],
        ["Data analysed", "2,539,608 training spectra of 275,810 different molecules, plus the sample test file"],
        ["Prepared", date.today().strftime("%d %B %Y")],
    ], [3.6 * cm, CONTENT_W - 3.6 * cm], header=False)
    st += [cover, Spacer(1, 1.0 * cm)]
    st.append(callout("How to read this document", [
        "Section 1 explains the competition goal with no chemistry background assumed. Section 2 explains how the "
        "measurements work. Section 3 describes the data. Section 4 lists the twelve most important things the data "
        "analysis revealed: each says <i>what we saw</i> and <i>why it matters</i>. Section 5 shows a first working "
        "baseline, and Section 6 gives the recommended plan. A glossary is at the end.",
        "All numbers come from the analysis code in the project repository (<i>scripts/eda/</i>). The document "
        "contains summary statistics only, because the competition rules do not allow sharing the raw data.",
    ]))
    st.append(PageBreak())

    # ===== Executive summary =====
    st.append(P("Summary in one page", "h1"))
    st.append(P("<b>The goal:</b> a lab instrument (a <i>mass spectrometer</i>) measures an unknown molecule by "
                "breaking it into pieces and weighing the pieces. From that list of weights, we must work out "
                "<b>which molecule it was</b>. For each mystery molecule we submit up to 25 guesses, best first, and "
                "we score more points the higher the right answer appears in our list."))
    st.append(P("<b>The most important things we found:</b>"))
    st += bullets([
        "<b>The sample test file is misleading.</b> All of its spectra are copies of training rows from the wrong "
        "type of chemistry (synthetic drug-like compounds). The real hidden test is natural products. We must build "
        "our own practice test instead.",
        "<b>No single training source looks like the test.</b> The biggest library (45% of the data) uses the same "
        "instrument as the test but contains the wrong kind of molecules. The public libraries contain the right kind "
        "of molecules but were measured on different instruments. A good solution must combine both.",
        "<b>A simple 'look it up' approach already works well for known molecules.</b> Matching the mystery spectrum "
        "against reference spectra, after filtering by weight, puts the right answer first 83% of the time "
        "(score 0.898 out of 1). Random guessing in the same shortlist scores 0.307.",
        "<b>The harder part is molecules nobody has measured before.</b> For those, a reference database of "
        "structures (such as PubChem) and a model that predicts structure from the spectrum will be needed.",
        "<b>The data needs careful cleaning.</b> Libraries differ hugely in how many peaks they keep, how precisely they "
        "record weights, and how they label energies. Some apparent label errors are real mistakes; one large group "
        "(50,785 spectra) only <i>looks</i> wrong because of a bug in one helper column.",
        "<b>Natural products leave tell-tale signs.</b> They often lose water and sugar units when broken. Those "
        "losses are 3–28 times more common than in the synthetic compounds, which gives us useful clues.",
    ])
    st.append(Spacer(1, 4))
    st.append(callout("Bottom line", [
        "Practise on the natural-product examples, not the sample test file. Clean every library the same way the "
        "hidden test was cleaned. Start with a weight filter plus spectrum matching, then add a structure database and "
        "learned models for the molecules that have never been measured.",
    ], wash=WASH_ORANGE, bar=ORANGE))
    st.append(PageBreak())

    # ===== 1. Competition =====
    st.append(P("1. What is the competition about?", "h1"))
    st.append(P("Nature is full of molecules: in plants, fungi, bacteria and our own bodies. Many could become "
                "medicines or tell doctors about disease, but most of them have never been identified. Scientists "
                "can <i>detect</i> thousands of molecules in a sample with a mass spectrometer, yet turning each "
                "measurement into a named chemical structure is slow and often impossible with today's tools."))
    st.append(P("Enveda, a company that searches nature for new medicines, is running this Kaggle competition to "
                "push the field forward. The name CASMI stands for <i>Critical Assessment of Small Molecule "
                "Identification</i>, a scientific challenge that ran from 2012 to 2022 and has now been revived."))
    st.append(P("The task in one sentence", "h2"))
    st.append(callout("", [
        "<b>Given the mass-spectrometry measurements of an unknown molecule, predict its chemical structure.</b> "
        "For each molecule, submit up to 25 candidate structures, ranked from most to least likely.",
    ]))
    st.append(P("What we submit", "h2"))
    st.append(P("A file with one row per mystery molecule. Each row lists up to 25 structures written as "
                "<b>SMILES</b> strings, a standard way to write a molecule as text (for example, ethanol is "
                "<font name='DV-B'>CCO</font>). The guesses are separated by semicolons, best guess first."))
    st.append(P("Submissions must be made from a Kaggle notebook that runs in at most 9 hours <b>with no internet "
                "access</b>. Anything our solution needs, such as reference spectra, structure databases or trained "
                "models, has to be uploaded to Kaggle beforehand as a dataset."))
    st.append(P("How the score works", "h2"))
    st.append(P("The score is called <b>MRR@25</b> (mean reciprocal rank). For each molecule we get "
                "<b>1 ÷ (the position of the first correct guess)</b>, or zero if the right answer is not in our 25. "
                "The final score is the average over all molecules, so it lies between 0 and 1."))
    st.append(scoring_drawing())
    st.append(P("Points earned for one molecule, depending on where the correct answer sits in our ranked list.",
                "caption"))
    st += bullets([
        "Getting the answer <b>first</b> matters a lot. Second place is already worth only half.",
        "A guess counts as correct if the molecule's atoms are connected the same way. The 3D arrangement "
        "(<i>stereochemistry</i>) and small hydrogen shifts (<i>tautomers</i>) are ignored, which makes the task "
        "fairer.",
        "Several spectra may belong to the same molecule (median 3, up to 16). We give <b>one</b> answer list per "
        "molecule, so we should combine the evidence from all of its spectra.",
    ])
    st.append(P("Three levels of difficulty", "h2"))
    st.append(P("Each hidden test molecule falls into one of three classes. The mix is secret."))
    st.append(table([
        ["Class", "What it means", "How it can be solved"],
        ["1: Known and measured", "Someone has already published a reference spectrum of this molecule.",
         "Look up the most similar reference spectrum (like a fingerprint database)."],
        ["2: Known, never measured", "The structure is listed in a chemical database (PubChem or COCONUT), but no "
         "spectrum exists.", "Search the structure database and predict which candidate fits the spectrum."],
        ["3: Brand-new", "The structure is not in any database.", "Build (generate) the structure from scratch."],
    ], [3.6 * cm, 6.6 * cm, CONTENT_W - 10.2 * cm]))
    st.append(Spacer(1, 6))
    st.append(P("The hidden test has about <b>1,500 spectra of about 400 molecules</b>, all measured on one "
                "instrument type (a Bruker <b>timsTOF</b>), with molecule weights between 157 and 1,159 daltons."))
    st.append(CondPageBreak(9 * cm))

    # ===== 2. Background =====
    st.append(P("2. How a mass spectrometer 'sees' a molecule", "h1"))
    st.append(P("A useful picture: imagine you are handed a bag of Lego pieces and must work out which model they "
                "came from. A mass spectrometer does something similar with molecules: it weighs the whole molecule, "
                "then smashes it and weighs the pieces."))
    st.append(pipeline_drawing())
    st.append(P("The result is a mass spectrum: a list of (piece weight, how much of it). The competition asks us "
                "to work backwards from that list to the molecule.", "caption"))
    st += bullets([
        "<b>Ionisation and adducts.</b> The instrument can only weigh charged things, so the molecule first picks up "
        "or loses a small charged partner, for example a proton (H<super>+</super>) or a sodium ion "
        "(Na<super>+</super>). This combination is called the <b>adduct</b>, written like "
        "<font name='DV-B'>[M+H]+</font> (molecule plus a proton). The test uses ten adduct types.",
        "<b>Precursor m/z.</b> The weight of the whole charged molecule, measured very precisely. It narrows down "
        "the possible chemical formula.",
        "<b>Collision energy.</b> How hard the molecule is smashed, in electron-volts (eV). Low energy leaves the "
        "molecule mostly intact; high energy breaks it into small pieces.",
        "<b>Spectrum and peaks.</b> The output is a list of fragment weights (<i>peaks</i>) and how abundant each one "
        "is. The tallest peak is the <b>base peak</b> and is scaled to 1.0.",
    ])
    st.append(KeepTogether([toy_spectrum(), P("A made-up example spectrum. Each bar is one fragment: position = "
                                              "its weight, height = how much of it was detected.", "caption")]))
    st.append(P("Different instruments (timsTOF, Orbitrap, Q-TOF, ion trap…) and settings produce noticeably "
                "different spectra for the same molecule, which is one of the main challenges in this competition."))
    st.append(CondPageBreak(9 * cm))

    # ===== 3. Data =====
    st.append(P("3. The data we were given", "h1"))
    st.append(P("Training data", "h2"))
    st.append(P("<b>2,539,608 spectra</b> covering <b>275,810 different molecules</b>, each labelled with the "
                "correct structure. They come from 11 source libraries, which were collected by different labs on "
                "different machines and processed in different ways:"))
    st.append(table([
        ["Library", "Spectra", "Molecules", "What it is"],
        ["enveda-180", "1,153,785", "182,941", "Enveda's own data. Same instrument as the test, but synthetic drug-like compounds"],
        ["pluskal_ms2 (MSnLib)", "527,581", "46,821", "Orbitrap spectra of commercial compound libraries"],
        ["riken", "347,171", "15,892", "Strong focus on plant metabolites"],
        ["gnps", "220,849", "45,750", "Largest community natural-product collection"],
        ["massbank", "101,727", "9,180", "Curated, many labs and instruments"],
        ["mona", "92,416", "11,681", "MassBank of North America"],
        ["spectraverse", "50,933", "9,631", "Harmonised mix of smaller public libraries"],
        ["msdial", "40,765", "9,127", "Libraries shipped with the MS-DIAL software"],
        ["drug_plus", "2,545", "2,539", "About one spectrum per drug"],
        ["enveda-np-examples", "1,184", "250", "<b>Closest to the test</b>: natural products, same instrument and processing"],
        ["masaryk", "652", "416", "Small set of chemical standards"],
    ], [3.9 * cm, 2.1 * cm, 2.3 * cm, CONTENT_W - 8.3 * cm]))
    st.append(Spacer(1, 6))
    st.append(P("Test data", "h2"))
    st.append(P("The test file we can download holds 1,213 spectra of 400 molecules, but it is only a "
                "<b>placeholder</b>. When a submission is scored, Kaggle swaps in the real hidden test set "
                "(about 1,500 spectra of about 400 natural-product molecules)."))
    st.append(figure("01_library_composition", "Figure 1. How many spectra (left) and different molecules (right) "
                     "each library contributes. enveda-180 dominates the training data."))
    st.append(CondPageBreak(9 * cm))

    # ===== 4. Findings =====
    st.append(P("4. What the data analysis revealed", "h1"))
    st.append(P("We analysed every spectrum and every molecule in four passes: the descriptive information, "
                "the peaks themselves, the chemistry of the molecules, and a first identification experiment. Below "
                "are the twelve findings that matter most, in plain terms."))

    st.append(finding(1, "The sample test file is not a fair practice test",
        "Every one of the 1,213 sample test spectra is an exact copy of a row in the enveda-180 training library. "
        "Those are synthetic, drug-like compounds. The real hidden test is natural products. The sample file is also "
        "not cleaned like the hidden test: 34.5% of its spectra still contain peaks heavier than the whole molecule, "
        "which were removed from the real test.",
        "Tuning a model on the sample file would give a misleadingly high score and steer us towards the wrong "
        "chemistry. We should practise on the 250 <i>enveda-np-examples</i> molecules instead, because they were "
        "measured and processed exactly like the hidden test.",
        wash=WASH_ORANGE, bar=ORANGE))
    st.append(Spacer(1, 8))

    st.append(finding(2, "Most molecules appear in only one library",
        "90% of the 275,810 molecules occur in just one library. The enveda-180 molecules (99% unique to it) "
        "almost never appear elsewhere. In contrast, every one of the 250 natural-product examples also appears in "
        "the public libraries, but none of them is in enveda-180.",
        "The public libraries are where test-like molecules live. enveda-180 is useful for learning how the test "
        "instrument behaves, but not for the kind of chemistry we will be asked about."))
    st.append(figure("02_library_overlap", "Figure 2. Share of each row library's molecules that also appear in each "
                     "column library. Dark = large overlap. enveda-180 (top row) overlaps with almost nothing.",
                     width=CONTENT_W * 0.78))

    st.append(finding(3, "Natural products are chemically very different from enveda-180",
        "We scored every molecule with a standard 'natural-product-likeness' measure (higher = more like a natural "
        "product). enveda-180 has a typical score of −1.5, the public libraries −0.4, and the natural-product "
        "examples +1.4. Every enveda-180 molecule contains nitrogen and 35% contain halogens (chlorine, fluorine…). "
        "In contrast, 52% of the natural-product examples contain only carbon, hydrogen and oxygen. Natural products "
        "also carry more oxygen and more 3D centres.",
        "A model trained mostly on enveda-180 will tend to guess nitrogen- and halogen-rich structures, which the "
        "test will punish. Training must lean on the public natural-product data."))
    st.append(figure("15_element_composition", "Figure 3. Which elements the molecules contain. Blue = enveda-180, "
                     "grey = public libraries, green = natural-product examples (test-like)."))
    st.append(figure("14_descriptor_distributions", "Figure 4. Six chemical properties compared. The green "
                     "natural-product curve sits far from the blue enveda-180 curve, especially for "
                     "natural-product-likeness (top middle)."))

    st.append(finding(4, "Close relatives of test-like molecules exist, but only in the public libraries",
        "For each natural-product example we found the most similar <i>different</i> molecule in the training data "
        "(similarity 0 = nothing in common, 1 = identical fingerprint). In the public libraries the closest relative "
        "has a typical similarity of 0.78, and 74% have a relative above 0.7. In enveda-180 it is only 0.28.",
        "Even for brand-new test molecules (class 3), we can probably find structurally similar 'cousins' in the "
        "public libraries, which is a good starting point for building the answer."))
    st.append(figure("17_nearest_analog_similarity", "Figure 5. How similar the closest relative in the training "
                     "data is. Green (public libraries) sits far to the right of blue (enveda-180).",
                     width=CONTENT_W * 0.8))

    st.append(finding(5, "Weight alone narrows things down, but not enough",
        "Only 9% of training molecules have a chemical formula no other molecule shares. A natural-product example "
        "shares its exact formula with about 6 other training molecules, and has about 13 training molecules within "
        "a 10 ppm weight window.",
        "Weight is a great first filter, but the fragment pattern is needed to choose between candidates. Against a "
        "full structure database such as PubChem there will be hundreds or thousands of candidates per weight, "
        "which is what makes classes 2 and 3 hard."))
    st.append(figure("16_formula_mass_ambiguity", "Figure 6. Left: how many training molecules share a formula. "
                     "Right: how many training molecules fall within a given weight tolerance."))

    st.append(finding(6, "The libraries record spectra very differently",
        "Spectra from the test-type instrument (timsTOF) keep many tiny peaks: a typical spectrum has about 140 "
        "peaks, and 75% of them are smaller than 0.1% of the tallest peak. Most public libraries ship only the "
        "bigger peaks (a typical 9–57 per spectrum). Some libraries round weights to two decimals "
        "(34% of MS-DIAL, 10% of MassBank), and 24–42% of spectra in several libraries contain impossible peaks "
        "heavier than the molecule itself.",
        "We must clean every library the same way before comparing or training: remove peaks heavier than the "
        "molecule, drop very small peaks, keep the strongest ones, and allow a small weight tolerance when matching."))
    st.append(figure("09_peaks_vs_intensity_floor", "Figure 7. Typical number of peaks per spectrum: all peaks "
                     "(grey), only peaks at least 0.1% of the tallest (blue), at least 1% (orange). A 1% cut-off "
                     "puts the libraries on a similar footing.", width=CONTENT_W * 0.92))
    st.append(figure("10_precursor_and_precision", "Figure 8. Left: how often the whole-molecule peak is visible. "
                     "Middle: spectra containing impossible peaks heavier than the molecule (zero for the test-like "
                     "examples). Right: spectra with rounded, low-precision weights."))

    st.append(finding(7, "How hard the molecule is smashed changes the spectrum a lot",
        "In enveda-180, the intact molecule is still visible in 98% of spectra at 20 eV, 54% at 40 eV, and only 5% "
        "at 60 eV. Higher energy also shifts the strongest peak towards smaller pieces. The test mixes 20, 40 and "
        "60 eV spectra, and a quarter are 'merged' spectra combining several energies. In the test file, "
        "negative-mode energies are written with a minus sign (−40), which simply needs flipping.",
        "Collision energy should be an input to any model, and combining a molecule's spectra taken at different "
        "energies gives a fuller picture: low energy confirms the weight, high energy reveals the building blocks."))
    st.append(figure("13_entropy_and_ce_effect", "Figure 9. Right: as energy rises (20 → 40 → 60 eV) the strongest "
                     "peak moves to smaller pieces (left on the axis). Left: natural products (green) spread their "
                     "signal over more pieces."))

    st.append(finding(8, "Natural products break in recognisable ways",
        "We measured how often the molecule loses common small units when broken. Compared with enveda-180, the "
        "natural-product examples lose water 3× more often, two waters 28× more often, and sugar units 6–10× more "
        "often (hexose sugar such as glucose: 162.05 Da; deoxyhexose such as rhamnose: 146.06 Da; pentose: 132.04 Da).",
        "These losses are simple, explainable clues. For example, a 162 Da loss strongly suggests a sugar is "
        "attached, which helps rank candidate structures."))
    st.append(figure("12_neutral_losses", "Figure 10. How often each small unit is lost per spectrum: natural-product "
                     "examples (green) vs enveda-180 (blue).", width=CONTENT_W * 0.88))

    st.append(finding(9, "Many test-relevant spectra lack a recorded energy or instrument",
        "GNPS, the largest community natural-product library (220,849 spectra), has no collision-energy values at "
        "all, nor do drug_plus and masaryk. Instruments are described by 82 different free-text spellings, which we "
        "grouped into six families. Only enveda-180 and the natural-product examples are timsTOF.",
        "Models must cope with missing energy information. Grouping instruments into families lets us choose which "
        "reference spectra to trust."))
    st.append(figure("05_instrument_family", "Figure 11. Instrument families per library after mapping the 82 "
                     "free-text names.", width=CONTENT_W * 0.9))

    st.append(finding(10, "Some labels are wrong, and one 'error' is not an error",
        "Each training spectrum records how far its measured weight is from the weight implied by its label. Large "
        "gaps cluster at exact values: 1.007 Da (one proton), 17.03 (ammonia), 18.01 (water) and 21.98 (sodium "
        "instead of hydrogen). These are mislabelled adducts in GNPS, MassBank and RIKEN (26.5% of RIKEN is off by "
        "more than 10 ppm). But in MSnLib all 50,785 'formate adduct' spectra show a 1.007 Da gap even though their "
        "weights are exactly right. The helper column is wrong, not the data.",
        "We can filter out or repair truly mislabelled spectra, but blindly filtering on that column would throw "
        "away about 10% of a high-quality library for no reason.", wash=WASH_ORANGE, bar=ORANGE))
    st.append(figure("07_precursor_error", "Figure 12. Left: share of spectra whose weight disagrees with the label by "
                     "more than 10 ppm. Right: the large disagreements are exact chemical offsets (vertical lines), "
                     "the signature of mislabelled adducts."))

    st.append(finding(11, "Some training spectra use adduct types the test never uses",
        "The training data uses 121 adduct types, while the test uses only 10. 29% of enveda-180 spectra are "
        "'dimers', two molecules stuck together, which never appear in the test. They produce a strong artificial "
        "peak at exactly half the measured weight.",
        "These spectra should be dropped or down-weighted when training for the test, or used only for general "
        "pre-training."))
    st.append(figure("04_adducts_by_library", "Figure 13. Adduct mix per library. Columns are the ten adducts used "
                     "in the test; the last column is everything else.", width=CONTENT_W * 0.9))

    st.append(finding(12, "The same molecule looks similar on modern instruments, but not on older ones",
        "For each natural-product example we compared its timsTOF spectrum with spectra of the same molecule from "
        "other instruments. The best match scores about 0.88 against Orbitrap and 0.86 against Q-TOF instruments "
        "(1 = identical), but close to 0 against ion-trap and triple-quadrupole instruments.",
        "Reference spectra from high-resolution instruments (Orbitrap, Q-TOF) are useful for matching against the "
        "test. Low-resolution libraries are much less useful for direct matching."))
    st.append(PageBreak())

    # ===== 5. Baseline =====
    st.append(P("5. A first baseline: look it up", "h1"))
    st.append(P("To learn how far a simple approach gets, we pretended the 250 natural-product examples were test "
                "molecules (they behave like class 1: reference spectra exist elsewhere) and ran a basic "
                "two-step search:"))
    st += bullets([
        "<b>Step 1, weigh:</b> from the measured weight and the adduct, compute the molecule's weight and keep every "
        "training molecule within 10 ppm. That leaves a typical shortlist of 13 (up to 227).",
        "<b>Step 2, compare fingerprints:</b> clean each spectrum, compare it with the reference spectra of every "
        "shortlisted molecule (from all other libraries), and rank the shortlist by best similarity, combining all of "
        "a molecule's spectra.",
    ])
    st.append(table([
        ["Result", "Value"],
        ["Score (MRR@25)", "<b>0.898</b> (out of 1.0)"],
        ["Right answer ranked 1st", "82.8% of molecules"],
        ["Right answer in the top 5", "98.4% of molecules"],
        ["Random order within the same shortlist", "0.307"],
    ], [8 * cm, CONTENT_W - 8 * cm]))
    st.append(Spacer(1, 6))
    st.append(figure("19_library_search_baseline", "Figure 14. Left: where the right answer was ranked for the 250 "
                     "molecules. Right: how similar the same molecule looks across instrument types."))
    st.append(callout("Important caveat", [
        "This is an optimistic number for the easiest class. The shortlist only included molecules from the training "
        "data. In the real test, classes 2 and 3 have no reference spectra, and searching a full database like "
        "PubChem gives far longer shortlists. The organisers also chose these 250 examples as common compounds, "
        "so they are easier than average. Expect the overall test score to be noticeably lower than 0.9.",
    ], wash=WASH_ORANGE, bar=ORANGE))
    st.append(PageBreak())

    # ===== 6. Recommendations =====
    st.append(P("6. What we recommend doing next", "h1"))
    st.append(P("Step 1: build an honest practice test", "h2"))
    st += bullets([
        "Use the 250 <i>enveda-np-examples</i> molecules as the main practice test, grouped by molecule like the real test.",
        "Imitate classes 2 and 3 by hiding some natural-product molecules completely: remove all of their spectra "
        "from the reference data, so the model has to identify them without a direct match.",
        "Never tune on the downloadable sample test file.",
    ])
    st.append(P("Step 2: clean all data the same way", "h2"))
    st += bullets([
        "Remove peaks heavier than the molecule (+2 Da), drop very small peaks (below 0.1–1% of the tallest), remove "
        "duplicate isotope peaks, keep the strongest ~64–128 peaks, and compress intensities (square root).",
        "Flip negative collision energies to positive and record whether a spectrum is merged from several energies.",
        "Drop or down-weight dimer and other non-test adducts; filter mislabelled spectra, except the MSnLib formate "
        "group, whose 'error' is a bookkeeping artefact.",
    ])
    st.append(P("Step 3: solve the three classes in layers", "h2"))
    st += bullets([
        "<b>Class 1 (known and measured):</b> weight filter + spectrum matching against high-resolution public "
        "references, combining all spectra of a molecule. Improve it with learned spectrum 'embeddings' (e.g. "
        "DreaMS, MS2DeepScore) adapted to timsTOF data.",
        "<b>Class 2 (known, never measured):</b> package an offline subset of PubChem/COCONUT within the test weight "
        "range, predict the formula first, then rank candidates with a model that scores how well a structure "
        "explains the spectrum (CSI:FingerID or MIST-style).",
        "<b>Class 3 (brand-new):</b> generate structures conditioned on the predicted formula and spectrum, starting "
        "from public natural-product relatives.",
        "<b>Use each data source for what it is good at:</b> enveda-180 for learning how the test instrument and "
        "energies behave; the public natural-product libraries for learning the right chemistry.",
    ])
    st.append(P("Step 4: respect the competition limits", "h2"))
    st += bullets([
        "Everything must run offline within 9 hours in a Kaggle notebook, so databases and models must be uploaded "
        "as Kaggle datasets in advance.",
        "Up to 5 submissions per day; choose 2 final submissions. Entry deadline: 7 December 2026.",
    ])
    st.append(PageBreak())

    # ===== Glossary =====
    st.append(P("Glossary", "h1"))
    gl = [
        ["Term", "Meaning in plain words"],
        ["Adduct", "The small charged partner a molecule gains or loses so it can be measured, e.g. [M+H]+ = molecule plus a proton."],
        ["Base peak", "The tallest peak in a spectrum; all other heights are given relative to it (it is 1.0)."],
        ["Collision energy (eV)", "How hard the molecule is smashed. Higher energy means smaller pieces."],
        ["Dalton (Da)", "The unit of molecular weight; one hydrogen atom weighs about 1 Da."],
        ["Fragment / peak", "One piece of the broken molecule, recorded as (weight, amount)."],
        ["Formula", "Which atoms, and how many, a molecule contains, e.g. C6H12O6 for glucose. Many structures share a formula."],
        ["InChIKey", "A fixed-length code identifying a molecule; its first block (14 letters) describes the atom connections and is used for scoring."],
        ["Library", "A published collection of reference spectra with known structures."],
        ["m/z", "Mass-to-charge ratio; for the singly charged ions here it is simply the weight."],
        ["MRR@25", "The competition score: average of 1/(rank of the first correct guess), counting only the top 25."],
        ["Natural product", "A molecule made by a living organism (plant, fungus, microbe, animal)."],
        ["Neutral loss", "The weight difference between the whole molecule and a fragment, i.e. the piece that broke off (e.g. water, a sugar)."],
        ["ppm", "Parts per million: a relative weight error. 10 ppm on 300 Da is 0.003 Da."],
        ["Precursor", "The intact charged molecule before smashing; its weight is measured first."],
        ["SMILES", "A way of writing a molecule's structure as a line of text, e.g. CCO for ethanol."],
        ["Spectrum", "The full list of fragment weights and amounts measured for one molecule at one setting."],
        ["Tanimoto similarity", "A 0–1 score of how similar two structures are (1 = same structural fingerprint)."],
        ["timsTOF / Orbitrap / Q-TOF", "Types of modern high-resolution mass spectrometer. The test uses timsTOF."],
    ]
    st.append(table(gl, [4.2 * cm, CONTENT_W - 4.2 * cm]))
    st.append(Spacer(1, 12))
    st.append(P("Where the numbers come from", "h2"))
    st.append(P("All figures and statistics were produced by the analysis scripts in the project repository "
                "(<i>scripts/eda/01_metadata.py</i> to <i>04_library_search.py</i>). The detailed technical report "
                "is <i>reports/eda/EDA_REPORT.md</i>, and the underlying tables are in <i>reports/eda/stats/</i>.",
                "small"))
    return st


def main():
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=1.8 * cm,
                            bottomMargin=2.0 * cm, title="Enveda CASMI 2026: Competition Guide and EDA Findings",
                            author="CASMI 2026 project", subject="Plain-language guide to the competition and data")
    doc.build(build(), onFirstPage=on_page, onLaterPages=on_page)
    print(OUT)


if __name__ == "__main__":
    main()
