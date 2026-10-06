"""Metadata helpers."""

import re


def instrument_family(s) -> str:
    if not isinstance(s, str) or s.strip().lower() in {"", "nan", "n/a", "none", "unknown"}:
        return "unknown"
    x = s.lower()
    if "tims" in x:
        return "timsTOF"
    if re.search(r"orbitrap|exactive|qft|itft|hybrid ft|\bft\b|fticr|lumos|fusion|velos", x):
        return "Orbitrap / FT"
    if re.search(r"tof|maxis|impact|synapt|xevo|tripletof", x):
        return "Q-TOF / TOF"
    if re.search(r"qqq|\bqq\b|-qq|quattro|triple|qtrap|tsq|api ?\d", x):
        return "Triple quad"
    if re.search(r"trap|\bit\b|-it\b|qit|lit|lcq|ltq", x):
        return "Ion trap"
    return "other"
