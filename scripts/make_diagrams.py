"""README diagram (light + dark): why assignment works as an instrument and why the naive comparison and the
pooled split are biased.  python scripts/make_diagrams.py -> docs/img/causal-{light,dark}.svg"""
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
THEMES = {
    "light": dict(bg="#ffffff", card="#f6f8fa", border="#d0d7de", ink="#1f2328", ink2="#59636e",
                  a="#2a78d6", a_soft="#e3eefb", b="#eb6834", b_soft="#fde8de", c="#1baf7a", c_soft="#dcf4ea"),
    "dark": dict(bg="#0d1117", card="#161b22", border="#30363d", ink="#e6edf3", ink2="#9198a1",
                 a="#3987e5", a_soft="#13263d", b="#d95926", b_soft="#3a1d12", c="#199e70", c_soft="#0f2e23"),
}
F = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
M = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def svg(t):
    p = []
    T = lambda x, y, s, size=13, col=None, w=400, a="start", mono=False: p.append(
        f'<text x="{x}" y="{y}" font-family="{M if mono else F}" font-size="{size}" font-weight="{w}" '
        f'fill="{col or t["ink"]}" text-anchor="{a}">{s}</text>')

    def node(cx, cy, w, h, title, sub, fill, stroke, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<rect x="{cx-w/2}" y="{cy-h/2}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.8"{d}/>')
        T(cx, cy - 2, title, 14, w=650, a="middle")
        T(cx, cy + 16, sub, 11.5, t["ink2"], a="middle")

    def arrow(x1, y1, x2, y2, col, label=None, lx=None, ly=None, dash=None, mk="g"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="2"{d} marker-end="url(#{mk})"/>')
        if label:
            T(lx, ly, label, 11.5, col, w=600, a="middle")

    T(32, 42, "What is randomized, what is not", 19, w=700)
    T(32, 64, "Only the assignment is random. Exposure is chosen by the ad auction, and the pooled split varies by advertiser.", 13, t["ink2"])
    node(150, 250, 190, 64, "assignment T", "85 / 15, random within a test", t["a_soft"], t["a"])
    node(470, 250, 190, 64, "exposure D", "3.6% of treated see an ad", t["card"], t["border"])
    node(790, 250, 190, 64, "visit / conversion Y", "outcome", t["card"], t["border"])
    node(630, 125, 210, 58, "user intent", "unobserved", t["b_soft"], t["b"], dash="5 4")
    node(150, 395, 210, 58, "advertiser / test", "unobserved in the data", t["b_soft"], t["b"], dash="5 4")
    node(470, 395, 190, 58, "features X (f0..f11)", "advertiser-dependent", t["card"], t["border"])
    arrow(245, 250, 371, 250, t["ink2"], "instrument", 308, 240)
    arrow(565, 250, 691, 250, t["c"], "effect we want", 628, 240, mk="c")
    arrow(590, 154, 520, 216, t["b"], "auction targets", 528, 176, mk="b")
    arrow(670, 154, 750, 216, t["b"], "intent drives Y", 760, 176, mk="b")
    arrow(150, 366, 150, 286, t["b"], "treatment ratio", 205, 330, dash="5 4", mk="b")
    arrow(255, 395, 371, 395, t["b"], None, mk="b")
    arrow(565, 380, 720, 285, t["ink2"], "X predicts Y", 680, 345)
    # callouts
    p.append(f'<rect x="900" y="104" width="160" height="96" rx="10" fill="{t["card"]}" stroke="{t["border"]}"/>')
    T(912, 126, "Naive exposed vs", 12, w=650); T(912, 143, "unexposed mixes the", 12); T(912, 160, "ad effect with intent", 12)
    T(912, 182, "→ 1.4 to 1.9x too big", 12, t["b"], w=700)
    p.append(f'<rect x="900" y="330" width="160" height="110" rx="10" fill="{t["card"]}" stroke="{t["border"]}"/>')
    T(912, 352, "Pooled split is not", 12, w=650); T(912, 369, "balanced on X: adjust", 12); T(912, 386, "for X (Lin, CUPAC,", 12)
    T(912, 403, "AIPW) before trusting", 12); T(912, 425, "the lift", 12, t["b"], w=700)
    defs = "".join(f'<marker id="{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                   f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for k, c in [("g", t["ink2"]), ("b", t["b"]), ("c", t["c"])])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1090 470" width="1090" height="470" role="img" '
            f'aria-label="Causal diagram: random assignment instruments ad exposure; user intent confounds exposure and outcome; '
            f'advertiser drives both treatment ratio and features."><defs>{defs}</defs>'
            f'<rect width="1090" height="470" rx="14" fill="{t["bg"]}" stroke="{t["border"]}"/>' + "".join(p) + "</svg>")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for k, t in THEMES.items():
        (OUT / f"causal-{k}.svg").write_text(svg(t))
    print("ok")
