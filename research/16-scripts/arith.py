"""Forensic arithmetic for 0.9906 (doc 16 §1). Exact expectations, no Monte Carlo.
Size prior from real train GT (10-frontier §1.1): identical across countries.
Per S1 with n true records: each true record is retrieved (blocking) w.p. q, then accepted w.p. a
(independent); FPs ~ Poisson(lam_n) (lam0 for singletons, lam for n>0).
F0.5 = 1.25k / (1.25k + 0.25(n-k) + f); pred empty & n>0 -> 0; n=0 -> 1 iff f=0.
"""
import numpy as np
from math import comb, exp, factorial

P = {0: .0558, 1: .0540, 2: .170, 3: .241, 4: .219, 5: .146, 6: .075, 7: .029, 8: .0085, 9: .0019, 10: .0003}
s = sum(P.values()); P = {k: v / s for k, v in P.items()}
W = {"IN": .4675, "US": .3827, "FR": .1498}   # test S1 shares (counted: 809,986 / 663,106 / 259,452)


def ef(n, r, lam, fmax=12):
    """E[F0.5] for an S1 of size n, per-true-record end-to-end recall r, Poisson(lam) FPs."""
    pf = [exp(-lam) * lam ** f / factorial(f) for f in range(fmax)]
    if n == 0:
        return pf[0]
    tot = 0.0
    for k in range(1, n + 1):
        pk = comb(n, k) * r ** k * (1 - r) ** (n - k)
        tot += pk * sum(pf[f] * 1.25 * k / (1.25 * k + 0.25 * (n - k) + f) for f in range(fmax))
    return tot


def macro(r, lam, lam0=None):
    lam0 = lam if lam0 is None else lam0
    return sum(P[n] * ef(n, r, lam if n else lam0) for n in P)


if __name__ == "__main__":
    print("mean size", sum(n * P[n] for n in P))
    # 1) oracle vs pair recall (independent misses, perfect matcher)
    print("\n[oracle(q)] independent-miss model")
    for q in [.95, .956, .96, .9714, .98, .985, .9905, .995, .998]:
        print(f"  PC={q:.4f} oracle={macro(q, 0):.5f}  loss/(1-PC)={(1 - macro(q, 0)) / (1 - q):.3f}")
    # empirical: v4 .9561->.9833 ; v10 IN .9714->.9898 ; Ayan .9905->.99715 (IN .99678)
    # 2) required seen score given France gap
    print("\n[seen required for LB 0.9906 given France gap g = F_seen - F_FR]")
    for g in [0, .005, .01, .015, .02, .03, .047]:
        fs = (0.9906 + W["FR"] * g)
        print(f"  g={g:.3f}: F_seen>={fs:.4f}, F_FR={fs - g:.4f}")
    # 3) iso-lines: which (r, lam) give 0.9906 / 0.990 / 0.985 / 0.975
    print("\n[pair recall r end-to-end x FP-per-S1 lam -> macro F0.5]")
    lams = [0, .005, .01, .02, .03, .05, .08]
    print("  r\\lam " + " ".join(f"{l:7.3f}" for l in lams))
    for r in [.97, .98, .985, .99, .993, .995, .997, 1.0]:
        print(f"  {r:.3f} " + " ".join(f"{macro(r, l):7.4f}" for l in lams))
