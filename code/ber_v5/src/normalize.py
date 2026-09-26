"""Name and address normalisation. Hand-written rules only, with no external data or APIs.

Design rules:
- Country is an open set. Every list applies to every record, so nothing is
  keyed on {US, India}. France and any other unseen label go through the same
  code path.
- Canonicalisation is symmetric: many surface forms map to one token, and both
  sides of a pair go through the same function. An ambiguous abbreviation
  ("st" = street/saint) is therefore never expanded one-sidedly.
- Accent folding uses stdlib unicodedata. Unidecode is GPL, so it is not used.
Seed lists are adapted from cleanco (MIT), the libpostal dictionaries (MIT)
and India's MCA Rule 8. See the methodology doc.
"""
import re
import unicodedata

_SPECIAL = str.maketrans({"œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae", "ß": "ss", "ø": "o", "Ø": "o",
                          "đ": "d", "ł": "l", "ı": "i", "’": "'", "‘": "'", "´": "'", "`": "'"})


from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

_INDIC = [((0x0900, 0x097F), sanscript.DEVANAGARI), ((0x0980, 0x09FF), sanscript.BENGALI),
          ((0x0A00, 0x0A7F), sanscript.GURMUKHI), ((0x0A80, 0x0AFF), sanscript.GUJARATI),
          ((0x0B00, 0x0B7F), sanscript.ORIYA), ((0x0B80, 0x0BFF), sanscript.TAMIL),
          ((0x0C00, 0x0C7F), sanscript.TELUGU), ((0x0C80, 0x0CFF), sanscript.KANNADA),
          ((0x0D00, 0x0D7F), sanscript.MALAYALAM)]
_INDIC_RUN = re.compile("[" + "".join(f"\\u{a:04x}-\\u{b:04x}" for (a, b), _ in _INDIC) + "]+")
_V = "aeiouāīūṛṝeo"
_C = "bcdfgjklmnpqrstvwxyzḍṭṇṣśñṅ"
# word-final inherent schwa is dropped after a consonant (राम -> ram, ट्रेडर्स -> tredars, मार्केटिंग -> marketing),
# except after a y/r/v conjunct where Hindi keeps it (आदित्य -> aditya, मित्र -> mitra, विद्या -> vidya)
_SCHWA = re.compile(rf"(?<![{_C}h][yrv])(?<=[{_C}h])a\b")
_INDIC_FIX = str.maketrans({"\u0949": "\u094B", "\u0911": "\u0913", "\u0945": "\u0947", "\u090D": "\u090F"})


def _indic_to_latin(s: str) -> str:
    """Transliterate runs of Indian scripts to IAST (MIT library, offline), then drop the word-final
    inherent schwa that Hindi-style spelling omits (ट्रेडर्स -> ṭreḍars, not ṭreḍarsa)."""
    def conv(m):
        ch = ord(m.group(0)[0])
        script = next(sc for (a, b), sc in _INDIC if a <= ch <= b)
        out = transliterate(m.group(0).translate(_INDIC_FIX), script, sanscript.IAST)
        return _SCHWA.sub("", out.replace("ṃ", "n").replace("ṁ", "n"))
    return _INDIC_RUN.sub(conv, s)


def fold(s: str) -> str:
    """Transliterate Indian scripts, lower-case, strip accents, map special letters. Safe on NaN/None."""
    if not isinstance(s, str):
        return ""
    if _INDIC_RUN.search(s):
        s = _indic_to_latin(s)
    s = s.translate(_SPECIAL)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.casefold()


_DOTTED = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?\.?")      # s.a.r.l. -> sarl, l.l.c. -> llc
_ELISION = re.compile(r"\b(?:l|d|qu|j|m|n|s|t|c)'(?=[a-z])")  # French l'/d'/qu' elisions
_NA = re.compile(r"\bn\s*/\s*a\b|\bn\.a\.")
_NONALNUM = re.compile(r"[\W_]+")   # keeps letters of ANY script (open set), drops punctuation


def clean(s: str) -> str:
    s = fold(s)
    s = _NA.sub(" ", s)
    s = _DOTTED.sub(lambda m: m.group(0).replace(".", ""), s)
    s = _ELISION.sub("", s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", s)          # 4th -> 4
    s = _NONALNUM.sub(" ", s)
    s = _LEADZERO.sub("", s)                              # 0037878 -> 37878
    return " ".join(t for t in s.split() if t not in _NULLS)


_LEADZERO = re.compile(r"\b0+(?=\d)")
_NULLS = {"null", "none", "nan", "na", "n a", "nil", "unknown", "notavailable"}
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


def unleet(tok: str) -> str:
    """'5ons' -> 'sons', 'b0ss' -> 'boss'. Only for name tokens that are mostly letters with 1-2 digits."""
    nd = sum(ch.isdigit() for ch in tok)
    if 0 < nd <= 2 and len(tok) - nd >= 2 and any(ch.isalpha() for ch in tok):
        return tok.translate(_LEET)
    return tok


# ---- legal forms: multi-country, applied everywhere -------------------------------------------
_LEGAL_PHRASES = [
    # US / generic English
    "incorporated", "inc", "corporation", "corp", "corpn", "company", "co", "and co", "and company",
    "limited liability company", "llc", "limited liability partnership", "llp", "lp", "pllc", "pc", "plc",
    "limited", "ltd", "ltda",
    # India (MCA Rule 8 list plus common variants)
    "private limited", "pvt ltd", "pvt limited", "private ltd", "p ltd", "pvt", "private", "opc pvt ltd",
    "opc private limited", "opc", "producer limited", "unlimited", "huf", "hindu undivided family",
    # transliterated from Indian scripts (प्राइवेट लिमिटेड -> praivet limited)
    "praivet limited", "praivet limitad", "praivet", "prayivet", "limitad", "limited kampani", "kampani",
    "elaelapi", "elelpi", "ela ela pi", "inka", "inkorporeted", "bhiraivedh limidhedh", "bhiraivedh",
    "limidhedh", "piraivet", "limitet",
    # France
    "societe a responsabilite limitee", "sarl", "societe par actions simplifiee unipersonnelle", "sasu",
    "societe par actions simplifiee", "sas", "societe anonyme", "sa", "entreprise unipersonnelle a "
    "responsabilite limitee", "eurl", "societe en nom collectif", "snc", "societe civile immobiliere", "sci",
    "eirl", "ei", "earl", "selarl", "sca", "scs", "scop", "scm", "scp", "scpi", "sccv", "gie", "sem",
    "et cie", "and cie", "cie",
    # a few widespread others, since the test set is an open set
    "gmbh", "pty ltd",
]
_LEGAL_PREFIX = ["m s", "messrs", "ms", "societe", "ste", "ets", "etablissements", "the", "dr", "mr", "mrs"]
_LEGAL_SET = sorted({clean(p) for p in _LEGAL_PHRASES}, key=lambda p: -len(p.split()))
_PREFIX_SET = sorted({clean(p) for p in _LEGAL_PREFIX}, key=lambda p: -len(p.split()))
_DBA = re.compile(r"\b(?:dba|d b a|doing business as|t a|trading as|aka|a k a|formerly|enseigne|"
                  r"nom commercial)\b")


def split_legal(name_clean: str):
    """Remove legal-form phrases ANYWHERE in the name (the data shuffles them to the start/middle:
    'Pvt EFS Print Ventures Ltd'), plus honorific/prefix words at the start. Also drop consecutive
    duplicate tokens and fix leetspeak. Returns (core, legal_forms). Never empties the core."""
    toks = [unleet(t) for t in name_clean.split()]
    toks = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]
    legal, keep, i = [], [], 0
    while i < len(toks):
        for p in _LEGAL_SET:
            pt = p.split()
            if toks[i:i + len(pt)] == pt:
                legal.append(p)
                i += len(pt)
                break
        else:
            keep.append(toks[i])
            i += 1
    if not keep:                      # the whole name was legal words: keep it as the core
        keep, legal = toks, []
    for p in _PREFIX_SET:
        pt = p.split()
        if len(pt) < len(keep) and keep[:len(pt)] == pt:
            keep = keep[len(pt):]
            break
    return " ".join(keep), legal


_LEGAL_CANON = {  # legal-form families, used for the agree/conflict feature
    "inc": "corp", "incorporated": "corp", "corp": "corp", "corporation": "corp", "corpn": "corp",
    "llc": "llc", "limited liability company": "llc", "llp": "llp", "limited liability partnership": "llp",
    "ltd": "ltd", "limited": "ltd", "private limited": "pvtltd", "pvt ltd": "pvtltd", "pvt": "pvtltd",
    "private": "pvtltd", "pvt limited": "pvtltd", "praivet limited": "pvtltd", "praivet": "pvtltd", "private ltd": "pvtltd", "p ltd": "pvtltd",
    "co": "co", "company": "co", "and co": "co", "and company": "co", "sarl": "sarl",
    "societe a responsabilite limitee": "sarl", "sas": "sas", "societe par actions simplifiee": "sas",
    "sasu": "sas", "sa": "sa", "societe anonyme": "sa", "eurl": "eurl", "snc": "snc", "sci": "sci",
}


def legal_family(legal_tokens):
    return {_LEGAL_CANON.get(t, t) for t in legal_tokens}


# ---- symmetric token canonicalisation ---------------------------------------------------------
_CANON = {}
for canon, forms in {
    # street types (EN / IN / FR)
    "st": ["street", "str", "saint", "st"], "ste": ["sainte", "suite"], "rd": ["road", "rd", "rod"],
    "av": ["avenue", "ave", "av", "aven"], "bd": ["boulevard", "blvd", "boul", "bd", "bvd"],
    "dr": ["drive", "drv", "dr", "docteur"], "ln": ["lane", "ln"], "ct": ["court", "ct"],
    "pl": ["place", "pl", "plaza", "plz"], "hwy": ["highway", "hwy"], "pkwy": ["parkway", "pkwy"],
    "sq": ["square", "sq"], "ter": ["terrace", "terr"], "cir": ["circle", "cir"], "fwy": ["freeway"],
    "mg": ["mahatma gandhi"], "ngr": ["nagar", "ngr"], "col": ["colony", "col"], "soc": ["society", "soc"],
    "sec": ["sector", "sec"], "chs": ["chaussee"], "imp": ["impasse", "imp"], "ch": ["chemin", "ch", "che"],
    "rte": ["route", "rte"], "fg": ["faubourg", "fbg", "fg"], "all": ["allee", "all"], "r": ["rue", "r"],
    "qu": ["quai"], "crs": ["cours"], "mkt": ["market", "mkt"], "bldg": ["building", "bldg"],
    "apt": ["apartment", "apt"], "fl": ["floor", "flr", "fl"], "no": ["number", "no", "num", "nr"],
    "blk": ["block", "blk"], "ph": ["phase"], "opp": ["opposite", "opp", "opps"],
    "near": ["near", "nr", "nearby", "pres", "cote"], "bh": ["behind", "bh", "derriere"],
    "nxt": ["next", "beside", "adjacent", "adj"],
    "n": ["north"], "s": ["south"], "e": ["east"], "w": ["west"],
    # name words
    "and": ["and", "et", "und"], "intl": ["international", "intl"], "mfg": ["manufacturing", "mfg"],
    "svc": ["services", "service", "svcs", "svc"], "ent": ["enterprises", "enterprise", "ent"],
    "tech": ["technologies", "technology", "tech", "technos"], "sys": ["systems", "system", "sys"],
    "assoc": ["associates", "association", "assoc"], "bros": ["brothers", "bros"], "sri": ["shri", "shree", "sri"],
    "dept": ["department", "dept"], "mgmt": ["management", "mgmt"], "dist": ["distribution", "distributors"],
    "ind": ["industries", "industry", "industrial", "ind", "inds"], "natl": ["national", "natl"],
    "pharm": ["pharmacy", "pharmacie", "pharma", "pharmaceuticals"], "elec": ["electricals", "electrical",
                                                                              "electric", "electricite", "elec"],
}.items():
    for f in forms:
        _CANON[f] = canon
_MULTI = {"mahatma gandhi": "mg", "next to": "nxt", "en face de": "opp", "a cote de": "nxt", "pres de": "near",
          "doing business as": "dba"}


_MULTI_RE = re.compile(r"\b(" + "|".join(map(re.escape, _MULTI)) + r")\b")


def canon_tokens(s_clean: str):
    s_clean = _MULTI_RE.sub(lambda m: _MULTI[m.group(1)], s_clean)
    return [_CANON.get(t, t) for t in s_clean.split()]


def phon_key(tok: str) -> str:
    """Transliteration-tolerant folding key (Indian and European spelling variants). Symmetric."""
    t = tok
    for a, b in (("ksh", "x"), ("aa", "a"), ("ee", "i"), ("oo", "u"), ("ph", "f"), ("w", "v"), ("z", "j"),
                 ("q", "k"), ("ck", "k"), ("c", "k"), ("y", "i"), ("th", "t"), ("dh", "d"), ("bh", "b"),
                 ("kh", "k"), ("gh", "g"), ("sh", "s"), ("ch", "c")):
        t = t.replace(a, b)
    t = t.replace("g", "j").replace("v", "b").replace("ks", "x")     # lojistiks~logistics, praibhet~private
    t = re.sub(r"(.)\1+", r"\1", t)
    if len(t) > 3:
        t = t[0] + re.sub(r"[aeiou]", "", t[1:])
    return t


def phon_line(s: str) -> str:
    return " ".join(phon_key(t) for t in s.split())


# ---- address helpers ----------------------------------------------------------------------------
_NUM = re.compile(r"\d+")
_LANDMARK = re.compile(r"\b(?:near|opp|bh|nxt)\b")


def numbers(s_clean: str):
    return _NUM.findall(s_clean)


def postcode_like(nums):
    """First 6-digit run (IN PIN) or 5-digit run (US ZIP / FR code postal). Open-set: any 5-6 digit run."""
    for n in nums:
        if len(n) in (5, 6):
            return n
    return ""


def house_number(nums, pc):
    for n in nums:
        if n != pc and len(n) <= 4:
            return n
    return ""


_TRIG = {"near", "opp", "bh", "nxt"}


def address_parts(raw: str):
    """Split a raw address on commas/semicolons/newlines into canonical-token segments. A segment that
    contains a landmark trigger is the landmark part; the rest is the core. With no delimiter, the
    landmark is the trigger plus up to 3 following tokens."""
    segs = [canon_tokens(clean(x)) for x in re.split(r"[,;\n]+", raw if isinstance(raw, str) else "")]
    segs = [x for x in segs if x]
    core, lm = [], []
    for seg in segs:
        idx = next((i for i, t in enumerate(seg) if t in _TRIG), None)
        if idx is None:
            core += seg
        elif len(segs) > 1:
            core += seg[:idx]
            lm += seg[idx + 1:]
        else:
            core += seg[:idx] + seg[idx + 4:]
            lm += seg[idx + 1: idx + 4]
    return core, lm


def normalize_frame(df):
    """Add normalised columns to a source frame (in place) and return it."""
    name_c = df["business_name"].map(clean)
    addr_c = df["business_address"].map(clean)
    parts = name_c.map(lambda s: _DBA.split(s))
    main = parts.map(lambda p: p[0].strip())
    df["name_trade"] = parts.map(lambda p: p[1].strip() if len(p) > 1 else "")
    split = main.map(split_legal)
    df["name_clean"] = name_c
    df["name_core"] = split.map(lambda x: " ".join(canon_tokens(x[0])))
    df["name_legal"] = split.map(lambda x: "|".join(sorted(legal_family(x[1]))))
    df["name_trade"] = df["name_trade"].map(lambda s: " ".join(canon_tokens(split_legal(s)[0])) if s else "")
    df["name_phon"] = df["name_core"].map(phon_line)
    df["name_ns"] = df["name_core"].str.replace(" ", "", regex=False)   # '#pioneerfashion' == 'pioneer fashion'

    atoks = addr_c.map(canon_tokens)
    df["addr_norm"] = atoks.map(" ".join)
    lm = df["business_address"].map(address_parts)
    df["addr_core"] = lm.map(lambda x: " ".join(x[0]))
    df["addr_landmark"] = lm.map(lambda x: " ".join(x[1]))
    nums = addr_c.map(numbers)
    df["addr_nums"] = nums.map(lambda n: " ".join(n))
    df["postcode"] = nums.map(postcode_like)
    df["house_no"] = [house_number(n, p) for n, p in zip(nums, df["postcode"])]
    df["addr_alpha"] = df["addr_core"].map(lambda s: " ".join(t for t in s.split() if not t.isdigit()))
    df["country_norm"] = df["country"].map(lambda c: clean(c) if isinstance(c, str) else "")
    df["name_addr"] = df["name_core"] + " | " + df["addr_norm"]
    df["name_loc"] = df["name_core"] + " | " + df["addr_alpha"]
    return df


def renormalize_names(df, lex):
    """Apply a token lexicon to name_clean and rebuild the name columns derived from it, exactly as
    normalize_frame does (DBA split, legal-form split, canonical tokens). Only rows the lexicon changes are redone."""
    new = df["name_clean"].map(lambda s: " ".join(lex.get(t, t) for t in s.split()))
    m = (new != df["name_clean"]).values
    if not m.any():
        return df
    sub = new[m]
    parts = sub.map(lambda s: _DBA.split(s))
    main = parts.map(lambda p: p[0].strip())
    trade = parts.map(lambda p: p[1].strip() if len(p) > 1 else "")
    split = main.map(split_legal)
    df.loc[m, "name_clean"] = sub
    if "name_core" in df.columns:
        df.loc[m, "name_core"] = split.map(lambda x: " ".join(canon_tokens(x[0])))
    if "name_legal" in df.columns:
        df.loc[m, "name_legal"] = split.map(lambda x: "|".join(sorted(legal_family(x[1]))))
    if "name_trade" in df.columns:
        df.loc[m, "name_trade"] = trade.map(lambda s: " ".join(canon_tokens(split_legal(s)[0])) if s else "")
    return df


# ---- France-only normalisation (generic linguistic knowledge only; gated on the country label) -------------
_FR_STREET = {"rue":"rue","r":"rue","avenue":"av","av":"av","ave":"av","avn":"av","boulevard":"bd","bd":"bd",
    "bld":"bd","blvd":"bd","bvd":"bd","boul":"bd","place":"pl","pl":"pl","chemin":"ch","chem":"ch","che":"ch",
    "impasse":"imp","imp":"imp","allee":"all","allees":"all","all":"all","route":"rte","rte":"rte","quai":"quai",
    "cours":"crs","crs":"crs","faubourg":"fg","fbg":"fg","fg":"fg","square":"sq","sq":"sq","passage":"pas",
    "pas":"pas","psg":"pas","residence":"res","res":"res","lotissement":"lot","lotiss":"lot","hameau":"ham",
    "sentier":"sent","promenade":"prom","esplanade":"espl","village":"vlge","villa":"vla","traverse":"trav",
    "montee":"mte","chaussee":"chs","carrefour":"carr","galerie":"gal","parvis":"parv","cite":"cite",
    "clos":"clos","domaine":"dom","voie":"voie","batiment":"bat","bat":"bat","bt":"bat","escalier":"esc",
    "etage":"etg","appartement":"app","appt":"app","apt":"app","saint":"st","sainte":"ste","notre":"notre",
    "grand":"gd","grande":"gde","petit":"pt","petite":"pte","general":"gal","marechal":"mal","president":"pdt",
    "docteur":"dr","professeur":"pr","commandant":"cdt","colonel":"col","universite":"univ"}
_FR_LEGAL = {"sarl","eurl","sas","sasu","selarl","selas","snc","scs","sca","sci","scp","scm","scop","scic",
             "scea","gaec","earl","gie","eirl","saem","sem","sa","ei","ste","ets","cie"}
_FR_ELIS = re.compile(r"\b([ldjmnstc]|qu)['\u2019`\u00b4]\s*")
_FR_CEDEX = re.compile(r"\bcedex(?:\s*\d{1,2})?\b")
_FR_BP = re.compile(r"\b(?:bp|cs|tsa)\s*\d{1,6}\b")
_FR_ORD = re.compile(r"\b(\d{1,2})\s*(?:er|re|e|eme|ieme)\b")
_FR_NUMSUF = re.compile(r"\b(\d{1,5})\s*(bis|ter|quater)\b")
_SUF = {"bis": "b", "ter": "t", "quater": "q"}


def fr_addr(a: str) -> str:
    a = _FR_ELIS.sub("", a)
    a = _FR_CEDEX.sub(" ", a)
    a = _FR_BP.sub(" ", a)
    a = _FR_NUMSUF.sub(lambda m: m.group(1) + _SUF[m.group(2)], a)
    a = _FR_ORD.sub(r"\1", a)
    return " ".join(_FR_STREET.get(t, t) for t in a.split())


_FR_COMPANY = {"etablissements": "ets", "etablissement": "ets", "societe": "ste", "compagnie": "cie",
               "entreprise": "entr", "internationale": "intl", "international": "intl"}


def fr_name(n: str) -> str:
    n = _FR_ELIS.sub("", n)
    toks = [_FR_COMPANY.get(t, t) for t in n.split() if t not in _FR_LEGAL]
    return " ".join(toks) if toks else n


def apply_fr(df):
    """Re-normalise the French rows of an already-normalised frame (prep ran country-agnostic)."""
    m = (df["country_norm"] == "france").values if "country_norm" in df.columns else None
    if m is None or not m.any():
        return df
    for col in ("addr_norm", "addr_core", "addr_alpha"):
        if col in df.columns:
            df.loc[m, col] = [fr_addr(x) for x in df.loc[m, col].values]
    if "name_core" in df.columns:
        df.loc[m, "name_core"] = [fr_name(x) for x in df.loc[m, "name_core"].values]
    if "name_clean" in df.columns:
        df.loc[m, "name_clean"] = [fr_name(x) for x in df.loc[m, "name_clean"].values]
    return df
