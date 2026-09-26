"""Synthetic dataset in the official layout. Used ONLY to smoke-test the pipeline.

It is never used to train the model that is submitted. The noise patterns
follow the problem statement: abbreviations, legal suffixes, DBA names, & vs
"and", word order, typos, landmarks, missing PIN/state and reordered address
parts. The look-alike cases follow the official video: a similar name at a
different address, and a different business at the same address.

Usage: python tools/make_synthetic.py --out /tmp/synth --n-train 3000 --n-test 2000
"""
import argparse
import os
import random

import pandas as pd

US = dict(
    cities=[("San Jose", "CA", "951"), ("Austin", "TX", "787"), ("Reno", "NV", "895"),
            ("Boise", "ID", "837"), ("Fargo", "ND", "581"), ("Denver", "CO", "802"),
            ("Seattle", "WA", "981"), ("Tampa", "FL", "336")],
    streets=["Market", "Oak", "Pine", "Elm", "Maple", "Lake", "Hill", "Main", "Cedar", "Washington",
             "Lincoln", "Park", "River", "Sunset", "Highland"],
    types=[("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Drive", "Dr"), ("Boulevard", "Blvd"),
           ("Lane", "Ln")],
    legal=[("Inc", "Incorporated", "Inc."), ("LLC", "L.L.C.", "LLC"), ("Corp", "Corporation", "Corp."),
           ("Co", "Company", "Co."), ("Ltd", "Limited", "Ltd.")],
    landmarks=["City Hall", "Central Library", "Union Station", "Memorial Hospital", "Walmart"],
    near=["Nr", "Near", "Next to", "Opp"],
)
IN = dict(
    cities=[("Mumbai", "Maharashtra", "400"), ("Pune", "Maharashtra", "411"), ("Bengaluru", "Karnataka", "560"),
            ("Chennai", "Tamil Nadu", "600"), ("Ranchi", "Jharkhand", "834"), ("Jaipur", "Rajasthan", "302"),
            ("Kolkata", "West Bengal", "700"), ("Hyderabad", "Telangana", "500")],
    streets=["MG", "Gandhi", "Nehru", "Station", "Linking", "Link", "Ring", "Tilak", "Sardar Patel", "Lal Bahadur"],
    types=[("Road", "Rd"), ("Marg", "Marg"), ("Nagar", "Ngr"), ("Street", "St"), ("Colony", "Col")],
    legal=[("Pvt Ltd", "Private Limited", "Pvt. Ltd."), ("Ltd", "Limited", "Ltd."), ("LLP", "LLP", "L.L.P."),
           ("& Co", "and Company", "& Co.")],
    landmarks=["SBI ATM", "Shani Mandir", "Bus Stand", "Railway Station", "Big Bazaar", "City Mall"],
    near=["Near", "Nr", "Opp", "Behind", "Beside"],
)
FR = dict(
    cities=[("Paris", "", "750"), ("Lyon", "", "690"), ("Marseille", "", "130"), ("Toulouse", "", "310"),
            ("Bordeaux", "", "330"), ("Lille", "", "590"), ("Nantes", "", "440")],
    streets=["de la République", "Victor Hugo", "Saint-Honoré", "Jean Jaurès", "des Lilas", "Pasteur",
             "du Général de Gaulle", "Sainte-Catherine", "de la Paix"],
    types=[("Rue", "R"), ("Avenue", "Av"), ("Boulevard", "Bd"), ("Place", "Pl"), ("Impasse", "Imp"),
           ("Chemin", "Ch")],
    legal=[("SARL", "S.A.R.L.", "Société à responsabilité limitée"), ("SAS", "S.A.S.", "SAS"),
           ("SA", "S.A.", "Société Anonyme"), ("EURL", "E.U.R.L.", "EURL"), ("SNC", "S.N.C.", "SNC")],
    landmarks=["Mairie", "Gare SNCF", "Hôtel de Ville", "Église Saint-Pierre"],
    near=["Près de", "En face de", "À côté de"],
)
CFG = {"US": US, "India": IN, "France": FR}

WORDS = {
    "US": ["Acme", "Delta", "Bright", "Summit", "Pioneer", "Evergreen", "Liberty", "Golden", "Blue Ridge",
           "Keystone", "Atlas", "Cascade", "Frontier", "Harbor", "Zen", "Kappa", "Nova", "Apex", "Redwood"],
    "India": ["Shree Ganesh", "Sai", "Laxmi", "Bharat", "Om", "Krishna", "Balaji", "Durga", "Mahalaxmi", "Surya",
              "Annapurna", "Vishwakarma", "Hindustan", "Sri Venkateswara", "Jai Hind", "Navrang", "Ashirwad"],
    "France": ["Dupont", "Lefèvre", "Boulangerie Martin", "Atelier Moreau", "Garage Bernard", "Café de Flore",
               "Maison Rousseau", "Pharmacie Laurent", "Librairie Girard", "Fromagerie Blanc", "Chez Léa"],
}
TRADES = {
    "US": ["Robotics", "Foods", "Cafe", "Traders", "Motors", "Bakery", "Logistics", "Dental", "Pharmacy",
           "Electric", "Plumbing", "Consulting", "Hardware", "Auto Repair"],
    "India": ["Traders", "Enterprises", "Textiles", "Electricals", "Medical Stores", "Sweets", "Motors",
              "Industries", "Jewellers", "Hardware", "Agencies", "Steel", "Pharma", "Foods"],
    "France": ["Distribution", "Services", "Immobilier", "Transports", "Conseil", "Bâtiment", "Traiteur",
               "Informatique", "Électricité", "Plomberie"],
}
TRANSLIT = {"Shree": ["Shri", "Sri"], "Laxmi": ["Lakshmi", "Luxmi"], "Krishna": ["Krsna", "Krishn"],
            "Jewellers": ["Jewelers"], "Sai": ["Saai"], "Bharat": ["Bharath"], "Surya": ["Suriya"],
            "Mahalaxmi": ["Mahalakshmi"], "Balaji": ["Baalaji"]}


def typo(s, rng):
    if len(s) < 5:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.choice("dsir")
    if op == "d":
        return s[:i] + s[i + 1:]
    if op == "s":
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op == "i":
        return s[:i] + rng.choice("aeiourstln") + s[i:]
    return s[:i] + rng.choice("aeiourstln") + s[i + 1:]


class Entity:
    def __init__(self, rng, country):
        c = CFG[country]
        self.country = country
        self.brand = rng.choice(WORDS[country])
        self.trade = rng.choice(TRADES[country])
        self.legal = rng.choice(c["legal"]) if rng.random() < 0.8 else None
        self.dba = f"{rng.choice(WORDS[country])} {rng.choice(TRADES[country])}" if rng.random() < 0.08 else None
        self.num = str(rng.randint(1, 999))
        self.street = rng.choice(c["streets"])
        self.stype = rng.choice(c["types"])
        self.city, self.state, pre = rng.choice(c["cities"])
        if country == "India":
            self.post = pre + f"{rng.randint(0, 999):03d}"
        elif country == "US":
            self.post = pre + f"{rng.randint(0, 99):02d}"
        else:
            self.post = pre + f"{rng.randint(1, 20):02d}"
        self.landmark = rng.choice(c["landmarks"])

    def name(self, rng, noisy):
        base = f"{self.brand} {self.trade}"
        if noisy:
            toks = base.split()
            toks = [rng.choice(TRANSLIT[t]) if t in TRANSLIT and rng.random() < 0.4 else t for t in toks]
            if rng.random() < 0.12 and len(toks) >= 2:
                toks = toks[1:] + toks[:1]
            base = " ".join(toks)
            if rng.random() < 0.18:
                base = typo(base, rng)
            if rng.random() < 0.3:
                base = base.upper() if rng.random() < 0.5 else base.lower()
            base = base.replace(" and ", " & ") if rng.random() < 0.5 else base.replace(" & ", " and ")
        if self.legal and (not noisy or rng.random() < 0.7):
            base = f"{base} {rng.choice(self.legal) if noisy else self.legal[0]}"
        if self.dba and noisy and rng.random() < 0.5:
            base = f"{base} DBA {self.dba}" if rng.random() < 0.5 else self.dba
        if self.country == "India" and noisy and rng.random() < 0.1:
            base = "M/s " + base
        return base

    def address(self, rng, noisy):
        c = CFG[self.country]
        stype = self.stype[1] if noisy and rng.random() < 0.5 else self.stype[0]
        if self.country == "France":
            street = f"{self.num} {stype} {self.street}"
            tail = f"{self.post} {self.city}"
            if noisy and rng.random() < 0.1:
                tail += f" CEDEX {rng.randint(1, 20)}"
            parts = [street, tail]
        else:
            street = f"{self.num} {self.street} {stype}"
            parts = [street, self.city]
            if self.state and (not noisy or rng.random() < 0.6):
                parts.append(self.state)
            if not noisy or rng.random() < 0.6:
                parts.append(self.post)
        if noisy:
            r = rng.random()
            if r < 0.1:
                return f"{rng.choice(c['near'])} {self.landmark}, {self.city}"
            if r < 0.25 and self.country == "India":
                parts.insert(1, f"{rng.choice(c['near'])} {self.landmark}")
            if rng.random() < 0.08:
                parts = parts[1:] + parts[:1]
            if rng.random() < 0.1:
                parts[0] = typo(parts[0], rng)
        return ", ".join(parts)


def build(rng, n_s1, countries, id_pool):
    s1, s2, s3, gt = [], [], [], []

    def nid(prefix):
        while True:
            i = f"{prefix}-{rng.randint(0, 999999):06d}"
            if i not in id_pool:
                id_pool.add(i)
                return i

    for _ in range(n_s1):
        country = rng.choice(countries)
        e = Entity(rng, country)
        sid = nid("S1")
        s1.append((sid, e.name(rng, False), e.address(rng, False), country))
        k = rng.choices([0, 1, 2, 3], weights=[35, 35, 20, 10])[0]
        matches = []
        for _ in range(k):
            dst, pre = (s2, "S2") if rng.random() < 0.55 else (s3, "S3")
            rid = nid(pre)
            dst.append((rid, e.name(rng, True), e.address(rng, True), country))
            matches.append(rid)
        gt.append((sid, ",".join(matches)))
        if rng.random() < 0.15:          # chain look-alike: similar name, different address
            f = Entity(rng, country)
            f.brand, f.trade, f.legal = e.brand, e.trade, e.legal
            dst, pre = (s2, "S2") if rng.random() < 0.5 else (s3, "S3")
            dst.append((nid(pre), f.name(rng, True), f.address(rng, True), country))
        if rng.random() < 0.15:          # different business, same address
            g = Entity(rng, country)
            g.num, g.street, g.stype, g.city, g.state, g.post = e.num, e.street, e.stype, e.city, e.state, e.post
            dst, pre = (s2, "S2") if rng.random() < 0.5 else (s3, "S3")
            dst.append((nid(pre), g.name(rng, True), g.address(rng, True), country))
    for _ in range(n_s1 // 3):           # records of businesses that are not in S1 at all
        country = rng.choice(countries)
        e = Entity(rng, country)
        dst, pre = (s2, "S2") if rng.random() < 0.5 else (s3, "S3")
        dst.append((nid(pre), e.name(rng, True), e.address(rng, True), country))
    cols = ["entity_id", "business_name", "business_address", "country"]
    frames = [pd.DataFrame(x, columns=cols).sample(frac=1, random_state=rng.randint(0, 10**6)) for x in (s1, s2, s3)]
    return frames, pd.DataFrame(gt, columns=["source1_entity_id", "matched_entity_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pool = set()
    for split, n, countries in (("train", a.n_train, ["US", "India"]), ("test", a.n_test, ["US", "India", "France"])):
        d = os.path.join(a.out, "dataset", split)
        os.makedirs(d, exist_ok=True)
        (f1, f2, f3), gt = build(rng, n, countries, pool)
        for i, f in enumerate((f1, f2, f3), 1):
            f.to_csv(os.path.join(d, f"{split}_source{i}.tsv"), sep="\t", index=False)
        name = "train_ground_truth.tsv" if split == "train" else "_hidden_test_ground_truth.tsv"
        gt.to_csv(os.path.join(d, name), sep="\t", index=False)
        print(split, len(f1), len(f2), len(f3), "singletons", (gt.matched_entity_ids == "").mean().round(3))


if __name__ == "__main__":
    main()
