# chart_hero.py — README 首图：HyperQwen 上游 / 170hx-fullstack / 本仓库最终方案，速度 + 精度（中英两版，1600px 宽）
# 版式：左侧速度（横向分组条，每行按该行最大值归一，条尾标 tok/s），右侧精度（4×3 色块表），底部一行口径说明。
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK JP"]

# ---- 数据（出处见 README「成绩摘要」与 reports/）
# A、B：reports/01 §2.1（同一张测试卡，2026-09-24）；开发场景 8 并发：reports/03（2026-09-25，单遍）
A = [159.2, 125.9, 631.0, 540.8, 121.1, 137.9]
B = [177.2, 101.0, 507.1, 364.5, 89.8, 104.6]
# 最终方案 = A ×（int8 输出头 / int4 输出头，reports/02 同卡）×（组合版 C1 / 基线 P，reports/04 同卡）
D2_A = [152.0 / 158.6, 126.2 / 126.4, 631.1 / 644.3, 510.2 / 533.9, 124.2 / 122.5, 136.0 / 137.9]
C1_P = [167.8 / 156.2, 136.7 / 126.3, 656.7 / 626.3, 541.7 / 511.4, 132.4 / 123.1, 134.9 / 130.4]
F = [a * x * y for a, x, y in zip(A, D2_A, C1_P)]
VALS = [A, B, F]

INK, SUB, MUTED, TRACK = "#1b1f24", "#57606a", "#8b949e", "#f3f4f6"
COL = ["#9AA0A6", "#E07B24", "#1F6FD1"]           # 上游 / fullstack / 本方案
GOOD, MID, BAD, NEU = "#E3F2E6", "#FDF1D3", "#FBE3E0", "#F3F4F6"

T = {
 "zh": dict(
  title="Qwen3.8-27B on CMP 170HX：三方对比",
  sub="同一张 CMP 170HX · 贪心解码 · 关思考 · 数据与口径见 reports/",
  names=["HyperQwen 上游", "170hx-fullstack 原方案", "本仓库最终方案"],
  sp_title="速度", sp_hint="tok/s，越高越好",
  rows=["单流 · 英文", "单流 · 中文", "8 并发 · 英文", "8 并发 · 中文", "32k 长输入", "开发场景 · 8 并发"],
  pr_title="精度", pr_hint="量化格式与相对误差，越低越好",
  cols=["HyperQwen\n上游", "170hx-\nfullstack", "本仓库"],
  prow=["主体权重", "输出头", "KV 缓存", "基座模型"],
  cells=[
   [("int4 对称", "11–16%", BAD), ("int4 非对称", "10–13%", BAD), ("int4 对称", "11–16%", BAD)],
   [("int4", "11–16%", BAD), ("bf16", "0.17%", GOOD), ("int8", "0.65–0.95%", GOOD)],
   [("bf16", "0.17%", GOOD), ("fp8", "2.65%", MID), ("bf16", "0.17%", GOOD)],
   [("官方", "", NEU), ("微调版", "评测差 <1σ", NEU), ("官方", "", NEU)],
  ],
  notes=["输出头误差降到上游的 ≈1/17", "投机解码改动不改变输出（贪心等价）"],
  foot="本仓库最终方案为同卡比值连乘推算（±3%）；170hx-fullstack 自报单流 218–300 tok/s，测试条件不同、本环境未复现。",
 ),
 "en": dict(
  title="Qwen3.8-27B on CMP 170HX: three setups compared",
  sub="Same CMP 170HX card · greedy decoding · thinking off · data and method in reports/",
  names=["HyperQwen upstream", "170hx-fullstack (original)", "This repo (final)"],
  sp_title="Speed", sp_hint="tok/s, higher is better",
  rows=["Single · EN", "Single · ZH", "8 conc. · EN", "8 conc. · ZH", "32k input", "Dev agent · 8 conc."],
  pr_title="Precision", pr_hint="quantization format and relative error, lower is better",
  cols=["HyperQwen\nupstream", "170hx-\nfullstack", "This repo"],
  prow=["Main weights", "LM head", "KV cache", "Base model"],
  cells=[
   [("int4 sym", "11–16%", BAD), ("int4 asym", "10–13%", BAD), ("int4 sym", "11–16%", BAD)],
   [("int4", "11–16%", BAD), ("bf16", "0.17%", GOOD), ("int8", "0.65–0.95%", GOOD)],
   [("bf16", "0.17%", GOOD), ("fp8", "2.65%", MID), ("bf16", "0.17%", GOOD)],
   [("official", "", NEU), ("fine-tuned", "evals within 1σ", NEU), ("official", "", NEU)],
  ],
  notes=["LM head error cut to ≈1/17 of upstream", "Speculative-decoding changes: greedy output unchanged"],
  foot="This repo (final) is derived by chaining two same-card ratios (±3%). 170hx-fullstack self-reports 218–300 tok/s single-stream; different setup, not reproduced here.",
 ),
}

def draw(lang):
    t = T[lang]
    W, H = 16, 9.2
    fig = plt.figure(figsize=(W, H), dpi=100, facecolor="white")
    fig.text(0.045, 0.945, t["title"], fontsize=30, weight="bold", color=INK, va="center")
    fig.text(0.045, 0.895, t["sub"], fontsize=14.5, color=SUB, va="center")
    # 图例 chips
    x = 0.045
    for n, c in zip(t["names"], COL):
        fig.patches.append(FancyBboxPatch((x, 0.833), 0.014, 0.022, boxstyle="round,pad=0,rounding_size=0.004",
                                          transform=fig.transFigure, color=c))
        tx = fig.text(x + 0.021, 0.844, n, fontsize=14.5, color=INK, va="center", weight="bold" if c == COL[2] else "normal")
        fig.canvas.draw(); bb = tx.get_window_extent().transformed(fig.transFigure.inverted())
        x = bb.x1 + 0.03

    # ---- 左：速度
    fig.text(0.045, 0.765, t["sp_title"], fontsize=20, weight="bold", color=INK, va="center")
    fig.text(0.045 + (0.05 if lang == "zh" else 0.065), 0.763, t["sp_hint"], fontsize=13, color=MUTED, va="center")
    ax = fig.add_axes([0.175, 0.105, 0.34, 0.62]); ax.set_xlim(0, 1.32); ax.axis("off")
    n = len(t["rows"]); rowh = 1.0; bh = 0.2
    ax.set_ylim(n * rowh - 0.35, -0.55)
    for i, lab in enumerate(t["rows"]):
        top = max(v[i] for v in VALS)
        y0 = i * rowh
        ax.text(-0.03, y0, lab, fontsize=14, color=INK, ha="right", va="center", clip_on=False)
        for k, v in enumerate(VALS):
            y = y0 + (k - 1) * (bh + 0.06)
            ax.barh(y, 1.0, height=bh, color=TRACK, zorder=0)
            ax.barh(y, v[i] / top, height=bh, color=COL[k], zorder=1)
            best = v[i] == top
            ax.text(v[i] / top + 0.02, y, f"{v[i]:.0f}", va="center", fontsize=12.5,
                    color=INK if k == 2 or best else SUB, weight="bold" if k == 2 else "normal")

    # ---- 右：精度
    X0 = 0.575
    fig.text(X0, 0.765, t["pr_title"], fontsize=20, weight="bold", color=INK, va="center")
    fig.text(X0 + (0.05 if lang == "zh" else 0.085), 0.763, t["pr_hint"], fontsize=13, color=MUTED, va="center")
    cx = [X0 + 0.09, X0 + 0.19, X0 + 0.29]; cw = 0.092
    for j, (c, name) in enumerate(zip(COL, t["cols"])):
        fig.patches.append(plt.Rectangle((cx[j], 0.705), cw, 0.006, transform=fig.transFigure, color=c))
        fig.text(cx[j] + cw / 2, 0.672, name, fontsize=12.5, ha="center", va="center", color=INK, weight="bold", linespacing=1.15)
    y = 0.622; rh = 0.098
    for i, rlab in enumerate(t["prow"]):
        fig.text(X0, y - rh / 2 + 0.006, rlab, fontsize=14, color=INK, va="center")
        for j, (fmt, err, bg) in enumerate(t["cells"][i]):
            fig.patches.append(FancyBboxPatch((cx[j] + 0.002, y - rh + 0.012), cw - 0.004, rh - 0.016,
                               boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure, color=bg))
            yc = y - rh / 2 + 0.004
            if err:
                fig.text(cx[j] + cw / 2, yc + 0.014, fmt, fontsize=12.5, ha="center", va="center", color=INK, weight="bold")
                fig.text(cx[j] + cw / 2, yc - 0.017, err, fontsize=12, ha="center", va="center", color=SUB)
            else:
                fig.text(cx[j] + cw / 2, yc, fmt, fontsize=12.5, ha="center", va="center", color=INK, weight="bold")
        y -= rh
    yn = y - 0.035
    for s in t["notes"]:
        fig.text(X0, yn, "✓", fontsize=14, color="#2E7D32", va="center", weight="bold")
        fig.text(X0 + 0.018, yn, s, fontsize=13.5, color=INK, va="center")
        yn -= 0.045

    # ---- 底部一行口径
    fig.lines.append(plt.Line2D([0.045, 0.955], [0.06, 0.06], transform=fig.transFigure, color="#e5e7eb", lw=1))
    fig.text(0.045, 0.03, t["foot"], fontsize=11.5, color=MUTED, va="center")
    fig.savefig(f"hero-{lang}.png", dpi=100, facecolor="white")
    plt.close(fig)

for lang in ("zh", "en"):
    draw(lang)
