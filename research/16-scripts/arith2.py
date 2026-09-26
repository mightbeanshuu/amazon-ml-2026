from arith import macro, W
# budget identity: LB = 1 - B - M - w_FR*g  (B, M = seen-country blocking / in-candidate matcher loss; FR = seen - g)
print("[required blocking loss B and PC for LB 0.9906]  ratio loss/(1-PC): 0.295 random-miss (Ayan-like), 0.357 ours (v10 IN measured)")
print("   M      g    B_max    PC_min(0.295)  PC_min(0.357)")
for M in [.003,.004,.005,.006,.007,.008,.0095,.012]:
    for g in [0,.01,.02]:
        B=0.0094-M-W['FR']*g
        s = f"{B:.4f}   " + (f"{1-B/.295:.4f}         {1-B/.357:.4f}" if B>0 else "impossible")
        print(f"  {M:.4f} {g:.3f}  {s}")
print("\n[matcher loss M = 1 - macro(r_m, lam) at perfect blocking]")
for r in [.97,.973,.98,.985,.99,.993,.995]:
    print(f"  r_m={r:.3f} " + "  ".join(f"lam={l:.3f}:M={1-macro(r,l):.4f}" for l in [.005,.01,.015,.02,.03]))
print("\n[singleton FP rate s -> loss 0.0558*s]", {s: round(.0558*s,4) for s in [.01,.02,.038,.047]})
# our v10 expected decomposition vs top-1
print("\nv4 IN in-candidate approx: r_m .973 lam .03 ->", round(1-macro(.973,.03),4))
