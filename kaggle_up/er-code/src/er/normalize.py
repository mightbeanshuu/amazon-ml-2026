"""Text normalization for business names and addresses.

Every function is country-agnostic: no hard-coded state lists or country rules, so
France (unseen in training) goes through exactly the same path as US/India.
Licenses: anyascii (ISC), stdlib re.
"""
import re
from anyascii import anyascii

_WS = re.compile(r"\s+")
_DOTTED = re.compile(r"\b(?:[a-z]\.\s?){2,}")                      # "l.l.c." -> "llc"
_NONALNUM = re.compile(r"[^a-z0-9 ]")
_PHONE_TAIL = re.compile(r"\s+-\s+\+?\d[\d\s-]{6,}$")          # "... - 7149969359"
_PIPE_TAIL = re.compile(r"\s*\|.*$")                              # "... | www.x.com"
_URL = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co|biz|info|fr)$")
_ALIAS = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|fka|f/k/a|formerly known as|doing business as|trading as|t/a)\b")
_LEADING_JUNK = re.compile(r"^[\W_]+")
_HONORIFIC = re.compile(r"^(?:m\s*/\s*s|messrs|shri|shree|sri|smt|mr|mrs|ms|the)\b\.?\s+")
_ORD = re.compile(r"\b(\d+)\s*(?:st|nd|rd|th)\b")
_LEAD0 = re.compile(r"\b0+(\d)")
_NULLTOK = re.compile(r"(?:^|(?<=[\s,]))(?:<?null>?|n/?a|none|nil|-+)(?=[\s,]|$)", re.I)
# leet: only inside tokens that also contain letters, so "15/58" or "2nd" stay numeric
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "8": "b", "@": "a", "$": "s"})

def to_ascii(s: str) -> str:
    return anyascii(s or "").lower()

def _deleet_token(t: str) -> str:
    return t.translate(_LEET) if re.search(r"[a-z]", t) and re.search(r"\d", t) else t

def clean_name(s: str) -> str:
    """Canonical name string: ascii, lowercase, junk/phone/url/alias stripped, leet folded."""
    s = to_ascii(s).strip()
    s = _PHONE_TAIL.sub("", s)
    s = _PIPE_TAIL.sub("", s)
    s = _LEADING_JUNK.sub("", s)
    m = _URL.match(s.replace(" ", ""))
    if m:
        s = m.group(1).replace("-", " ")
    s = _DOTTED.sub(lambda m: m.group(0).replace(".", "").replace(" ", "") + " ", s)
    parts = _ALIAS.split(s)            # "X dba Y": keep both sides, separated
    s = " ".join(parts)
    s = _HONORIFIC.sub("", s)
    s = " ".join(_deleet_token(t) for t in s.split())
    s = s.replace("&", " and ").replace("+", " and ")
    s = _NONALNUM.sub(" ", s)
    toks = s.split()
    out = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]   # "llc llc" -> "llc"
    return " ".join(out)

def clean_address(s: str) -> str:
    s = to_ascii(s)
    s = _NULLTOK.sub(" ", s)
    s = _DOTTED.sub(lambda m: m.group(0).replace(".", "").replace(" ", "") + " ", s)
    s = s.replace("#", " ")
    s = _ORD.sub(r"\1", s)
    s = _LEAD0.sub(r"\1", s)
    s = _NONALNUM.sub(" ", s)
    return _WS.sub(" ", s).strip()

_VOW = re.compile(r"[aeiouy]")
_SKEL_MAP = [("ph", "f"), ("w", "v"), ("q", "k"), ("c", "k"), ("z", "s"), ("x", "ks")]

def skeleton(s: str) -> str:
    """Consonant skeleton per token — bridges transliteration ('inovetiv pavr' ~ 'innovative power')."""
    out = []
    for t in s.split():
        if t.isdigit():
            continue
        for a, b in _SKEL_MAP:
            t = t.replace(a, b)
        t = t[0] + _VOW.sub("", t[1:]) if t else t
        t = re.sub(r"(.)\1+", r"\1", t)
        out.append(t)
    return " ".join(out)

def numbers(s: str) -> frozenset:
    return frozenset(re.findall(r"\d+", s))
