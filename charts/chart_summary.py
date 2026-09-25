# chart_summary.py — 手机长图：170HX 上 Qwen3.8-27B 推理优化总结（宽 1080px）
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK JP"]

W, H = 1080, int(sys.argv[1]) if len(sys.argv) > 1 else 5930
OUT = sys.argv[2] if len(sys.argv) > 2 else "summary-long.png"
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")

INK, SUB, MUTED = "#1a1a1a", "#444", "#777"
GREEN, GREEN_L, RED, BLUE, ORANGE, GREY, PURPLE = "#2E7D32", "#A5D6A7", "#C62828", "#1565C0", "#E65100", "#9E9E9E", "#6A1B9A"
L, R = 60, 1020  # 左右边距（px）


def fy(y):  # 自顶向下的像素 → figure 坐标
    return 1 - y / H


def text(x, y, s, size=16, color=INK, weight="normal", ha="left", va="top", ls=1.5):
    fig.text(x / W, fy(y), s, fontsize=size, color=color, weight=weight, ha=ha, va=va, linespacing=ls)


def box(x, y, w, h, fc, ec="none", r=18):
    fig.patches.append(FancyBboxPatch((x / W, fy(y + h)), w / W, h / H, boxstyle=f"round,pad=0,rounding_size={r / W}",
                                      transform=fig.transFigure, fc=fc, ec=ec, lw=1.5, zorder=0))


def axes(x, y, w, h):
    return fig.add_axes([x / W, fy(y + h), w / W, h / H])


def section(y, num, title, color):
    fig.patches.append(Rectangle((L / W, fy(y + 44)), 10 / W, 44 / H, transform=fig.transFigure, color=color))
    text(L + 26, y + 2, f"{num}  {title}", size=27, weight="bold")
    return y + 64


def wrap(s, width):  # 按显示宽度折行：CJK 记 1，ASCII 记 0.55
    out, cur, w = [], "", 0.0
    for ch in s:
        cw = 0.6 if ord(ch) < 0x2E80 else 1.0
        if w + cw > width and cur:
            out.append(cur); cur, w = "", 0.0
        cur += ch; w += cw
    return out + ([cur] if cur else [])


def clean_ax(ax, keep=("left",)):
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(s in keep)
    ax.tick_params(length=0)


# ---------------- 标题
y = 50
text(W / 2, y, "170HX 上跑 Qwen3.8-27B", size=40, weight="bold", ha="center")
text(W / 2, y + 62, "推理优化全记录：三项上线，七项否决", size=28, weight="bold", ha="center", color=GREEN)
text(W / 2, y + 112, "CMP 170HX 64GB · 单卡单副本 · HyperQwen（vLLM + MTP） · 2026-09", size=17, color=MUTED, ha="center")
y = 230
box(L, y, R - L, 190, "#F1F8E9")
text(L + 30, y + 22, "最终效果（生产同时段 A/B，组合版 vs int8 输出头版）", size=19, weight="bold", color=GREEN)
for i, (k, v) in enumerate([("单流 英文", "+12%"), ("单流 中文", "+11%"), ("8 并发 英文", "+8%"), ("8 并发 中文", "+6%")]):
    cx = L + 30 + i * 235
    text(cx + 100, y + 70, v, size=40, weight="bold", color=GREEN, ha="center")
    text(cx + 100, y + 130, k, size=17, color=SUB, ha="center")

# ---------------- 1 硬件与基线
y = section(470, "1", "硬件与基线", BLUE)
rows = [
    ("GPU", "NVIDIA CMP 170HX（GA100，sm_80），每卡跑一个独立副本"),
    ("显存", "64GB 需 cmpunlocker 解锁驱动；驱动升级后必须重打补丁"),
    ("互联", "PCIe Gen2 x4，无 P2P；满载时功耗墙 250W"),
    ("模型", "Qwen3.8-27B W4A16（AutoRound int4 g128，Marlin 内核）"),
    ("投机", "MTP 起草 k=4，全词表起草（MTP_DRAFT_VOCAB=0）"),
    ("部署", "7 个副本分布在 2 个节点，前面是 litellm 网关"),
    ("起点", "vLLM 0.28：单流 英 159 / 中 126，8 并发 631 / 541 tok/s"),
]
for i, (k, v) in enumerate(rows):
    yy = y + i * 50
    text(L + 10, yy, k, size=18, weight="bold", color=BLUE)
    text(L + 110, yy, v, size=18, color=INK)
y += len(rows) * 50 + 30

# ---------------- 2 已上线
y = section(y, "2", "已上线的三项", GREEN)

# 2.1 int8 输出头
text(L, y, "① 输出头 int4 → int8", size=23, weight="bold")
text(L, y + 42, "只把 lm_head 换回基座自带的 int8，其余不动", size=17, color=SUB)
ax = axes(L + 170, y + 90, 520, 280)
cats = ["单流 英文", "单流 中文", "8 并发 英文", "8 并发 中文", "4k 长输入", "32k 长输入"]
d = [-4.2, -0.1, -2.0, -4.4, -7.2, 1.3]
yy = np.arange(len(cats))
ax.barh(yy, d, color=[RED if v < 0 else GREEN for v in d], height=0.6)
for a, v in zip(yy, d):
    ax.text(v + (-0.3 if v < 0 else 0.3), a, f"{v:+.1f}%", va="center", ha="right" if v < 0 else "left", fontsize=15, weight="bold")
ax.axvline(0, color="#333", lw=1); ax.set_xlim(-10, 4); ax.set_xticks([])
ax.set_yticks(yy); ax.set_yticklabels(cats, fontsize=15); ax.invert_yaxis(); clean_ax(ax, ())
text(L + 700, y + 100, "速度代价", size=17, color=MUTED)
text(L + 700, y + 128, "常规负载 慢 0~5%\n4k 长输入 慢 7%", size=19, weight="bold", color=RED)
text(L + 700, y + 220, "与 int4 版逐 token 比对", size=17, color=MUTED)
text(L + 700, y + 248, "top1 一致率 97.6%\n（测法噪声底 99.45%）", size=19, weight="bold", color=GREEN)
y += 395
text(L, y, "约每 55 个 token 有 1 处选词不同，都落在 top1 概率 <~0.6 的「分岔 token」上。",
     size=16, color=SUB)
y += 70

# 2.2 组合版
text(L, y, "② 组合版：vLLM 0.30 + int4 起草头", size=23, weight="bold")
text(L, y + 42, "vLLM 0.30（HyperQwen PR#189 自建镜像）；MTP 起草读 int4 输出头，验证仍用 int8", size=17, color=SUB)
ax = axes(L + 170, y + 95, 700, 330)
cats = ["单流 英文", "单流 中文", "8 并发 英文", "8 并发 中文"]
base = [151.9, 123.4, 608.9, 511.0]; combo = [170.4, 137.4, 659.5, 540.2]
ax.set_xlim(0, 1.32)
yy = np.arange(4); bh = 0.36
for i, (b, c) in enumerate(zip(base, combo)):
    ax.barh(i - bh / 2, 1.0, height=bh * 0.92, color=GREY)
    ax.barh(i + bh / 2, c / b, height=bh * 0.92, color=GREEN)
    ax.text(1.0 + 0.01, i - bh / 2, f"{b:g}", va="center", fontsize=14, color=SUB)
    ax.text(c / b + 0.01, i + bh / 2, f"{c:g}  ({c / b - 1:+.0%})", va="center", fontsize=15, weight="bold", color=GREEN)
ax.set_yticks(yy); ax.set_yticklabels(cats, fontsize=15); ax.invert_yaxis(); ax.set_xticks([]); clean_ax(ax, ())
fig.patches.append(Rectangle(((L + 170) / W, fy(y + 450)), 26 / W, 16 / H, transform=fig.transFigure, color=GREY))
text(L + 204, y + 440, "int8 输出头版（当时的生产配置）", size=15, color=SUB)
fig.patches.append(Rectangle(((L + 580) / W, fy(y + 450)), 26 / W, 16 / H, transform=fig.transFigure, color=GREEN))
text(L + 614, y + 440, "组合版（tok/s，3 轮中位）", size=15, color=SUB)
y += 490
box(L, y, R - L, 150, "#F5F5F5")
text(L + 24, y + 18, "• 独立测试卡基准：单流 +7.4 / +8.3%，8 并发 +4.9 / +5.9%，长输入 +8~9%", size=16)
text(L + 24, y + 52, "• 贪心输出与基线等价（分歧点 top1-top2 差 ≤0.25 nats）", size=16)
text(L + 24, y + 86, "• 金丝雀浸泡 80 分钟：319 个请求，0 错误，0 Xid，0 次重启", size=16)
text(L + 24, y + 116, "• 2026-09-25 滚动上线到全部 7 个副本", size=16, weight="bold", color=GREEN)
y += 190

# 2.3 会话粘滞
text(L, y, "③ 网关会话粘滞（litellm session_affinity）", size=23, weight="bold")
text(L, y + 42, "同一会话固定打到同一副本，吃满前缀缓存；两副本、8 条 agent 会话回放，两轮对照", size=17, color=SUB)
for j, (title, sh, st, unit) in enumerate([("后续轮首字延迟（秒）", [9.46, 4.37], [1.45, 2.45], "s"),
                                            ("每条会话耗时（秒）", [100, 89], [37, 53], "s")]):
    ax = axes(L + 90 + j * 470, y + 130, 380, 230)
    x = np.arange(2); bw = 0.36
    ax.bar(x - bw / 2, sh, bw * 0.92, color=GREY); ax.bar(x + bw / 2, st, bw * 0.92, color=GREEN)
    for a, v in zip(x - bw / 2, sh): ax.text(a, v, f"{v:g}", ha="center", va="bottom", fontsize=14, color=SUB)
    for a, v in zip(x + bw / 2, st): ax.text(a, v, f"{v:g}", ha="center", va="bottom", fontsize=15, weight="bold", color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(["第 1 轮", "第 2 轮"], fontsize=15); ax.set_yticks([])
    ax.set_ylim(0, max(sh) * 1.2); clean_ax(ax, ("bottom",))
    text(L + 90 + j * 470, y + 90, title, size=18, weight="bold")
fig.patches.append(Rectangle(((L + 170) / W, fy(y + 408)), 26 / W, 16 / H, transform=fig.transFigure, color=GREY))
text(L + 204, y + 398, "随机分发（simple-shuffle）", size=15, color=SUB)
fig.patches.append(Rectangle(((L + 500) / W, fy(y + 408)), 26 / W, 16 / H, transform=fig.transFigure, color=GREEN))
text(L + 534, y + 398, "会话粘滞", size=15, color=SUB)
y += 440
text(L, y, "会话 id 取自 x-litellm-session-id 或 x-*-session-id；不带 id 的请求仍随机分发。", size=16, color=SUB)
y += 80

# ---------------- 3 精度损失对比
y = section(y, "3", "精度损失对比", PURPLE)
text(L, y, "teacher forcing：把基线的贪心输出喂给被测配置，逐位置比较 top1 是否相同", size=16, color=SUB)
y += 44
items = [  # (名称, 一致率 %, 档位)
    ("噪声底：同配置自比", 99.51, "noise"),
    ("int4 起草头（组合版的一半）", 99.48, "eq"),
    ("MTP + suffix 混合（τ=4）", 99.52, "eq"),
    ("KV int8", 99.46, "eq"),
    ("KV fp8", 99.20, "lossy"),
    ("KV fp8 · 4k 长输入", 98.44, "lossy"),
    ("int8 输出头 vs int4 输出头", 97.63, "fix"),
]
TIER = {"noise": GREY, "eq": GREEN, "lossy": RED, "fix": BLUE}
ax = axes(L + 330, y + 10, 540, 420)
yy = np.arange(len(items))
ax.axvspan(99.45, 99.51, color="#FFE082", alpha=0.7, zorder=0)
ax.barh(yy, [v - 97.0 for _, v, _ in items], left=97.0, height=0.62, color=[TIER[t] for *_, t in items], zorder=2)
for a, (_, v, t) in zip(yy, items):
    ax.text(max(v, 99.51) + 0.06, a, f"{v:.2f}%", va="center", fontsize=15, weight="bold", color=TIER[t])
ax.set_xlim(97.0, 100.2); ax.set_ylim(len(items) - 0.5, -0.5)
ax.set_yticks(yy); ax.set_yticklabels([n for n, *_ in items], fontsize=15)
ax.set_xticks([97, 98, 99, 100]); ax.set_xticklabels(["97%", "98%", "99%", "100%"], fontsize=13, color=MUTED)
clean_ax(ax, ("bottom",))
y += 470
for i, (c, lab) in enumerate([(GREY, "噪声底 99.45~99.51%（黄带）"), (GREEN, "无损 / 等价"), (RED, "有损"), (BLUE, "纠正 int4 头的误差（理论推算）")]):
    xx = L + (i % 2) * 470; y2 = y + (i // 2) * 34
    fig.patches.append(Rectangle(((xx + 10) / W, fy(y2 + 20)), 26 / W, 16 / H, transform=fig.transFigure, color=c))
    text(xx + 46, y2 + 2, lab, size=15, color=SUB)
y += 84
for ln in ["• 组合版 = vLLM 0.30 + int4 起草头：贪心分歧全是近平局（≤0.25 nats）。起草头只决定",
           "   草稿，最终 token 仍由 int8 头验证，所以无损，只影响速度。",
           "• int8 输出头改掉的约 1.8% 位置，都是 top1 概率 <~0.6 的近平局 token。",
           "• 基线：int8 头一行对 int4 版（噪声底 99.45%），其余对当时的生产配置（99.51%）。"]:
    text(L, y, ln, size=15, color=SUB); y += 30
y += 26
text(L, y, "量化理论误差（group=128 数值模拟，误差 RMS / 原值 RMS）", size=19, weight="bold")
y += 44
fmt = [("int4 对称", "11~16%", RED), ("int4 非对称", "10~13%", RED), ("fp8（KV）", "2.65%", ORANGE),
       ("int8", "0.65~0.95%", GREEN), ("bf16", "0.17%", GREEN)]
for i, (k, v, c) in enumerate(fmt):
    xx = L + 10 + (i % 3) * 320; y2 = y + (i // 3) * 40
    text(xx, y2, k, size=16, color=SUB); text(xx + 170, y2, v, size=16, weight="bold", color=c)
y += 94
box(L, y, R - L, 250, "#F3E5F5")
text(L + 24, y + 16, "最终上线配置的精度来源", size=18, weight="bold", color=PURPLE)
src = [("主体权重", "int4 对称 g128（AutoRound 校准）"), ("输出头", "int8：误差从 11~16% 降到 0.65~0.95%"),
       ("起草头", "int4：只影响接受率和速度，不影响输出"), ("KV 缓存", "bf16：与原模型的注意力状态一致")]
for i, (k, v) in enumerate(src):
    text(L + 24, y + 60 + i * 44, k, size=16, weight="bold", color=PURPLE)
    text(L + 150, y + 60 + i * 44, v, size=16)
y += 280
for ln in ["• 对比 170hx-fullstack 原方案：它用非对称 int4、bf16 输出头、fp8 KV，理论保真度略高，",
           "   差距约 1–2 dB。它的微调模型：GPQA +3.5、MMLU −1.8、LCB +5，都在 1σ 以内，不显著。"]:
    text(L, y, ln, size=15, color=SUB); y += 30
y += 50

# ---------------- 4 否决
y = section(y, "4", "否决的方案", RED)
rej = [
    ("170hx-fullstack 原方案", "SGLang + DFlash2", "8 并发慢 19~33%\n开发场景慢 15~24%", "英文单流更快；本负载下 DFlash2\n草稿的中文接受率约 23%"),
    ("截断草稿词表", "32k / 48k / 64k", "慢 3~17%", "覆盖缺口让接受长度掉 0.1~0.4，\n省下的读取量补不回来"),
    ("KV 缓存 fp8", "", "慢 25~59%", "sm80 只能走 FlashInfer，CUDA graph\n退化成分段模式；精度也低于噪声底"),
    ("KV 缓存 int8", "", "开发场景\n慢 53~59%", "Triton 预填充太慢，\n长前缀首字延迟翻倍"),
    ("cpuset 绑核 / nice", "", "±1%", "瓶颈在 GPU 功耗墙，不在 CPU 调度"),
    ("MTP + suffix 混合", "K=4", "开发场景单会话\n慢 12%", "长上下文里 4-gram 常匹配到错误续写；\n每步还要等采样结果拷回 CPU"),
    ("DFlash2 on vLLM sm80", "", "Xid 31 崩溃", "上游未修；目前唯一经实测的修法是\nwtdcode 分支 PR 82 的边界 mask"),
]
cw = [285, 245, 430]
text(L + 14, y, "方案", size=16, weight="bold", color=MUTED)
text(L + 14 + cw[0], y, "结果（相对基线）", size=16, weight="bold", color=MUTED)
text(L + 14 + cw[0] + cw[1], y, "原因", size=16, weight="bold", color=MUTED)
y += 40
for i, (a, a2, b, c) in enumerate(rej):
    lines = c.split("\n")
    n = max(len(lines), b.count("\n") + 1, 2 if a2 else 1)
    rh = 30 * n + 34
    if i % 2 == 0:
        box(L, y - 12, R - L, rh, "#FFF5F5", r=10)
    text(L + 14, y, a, size=17, weight="bold")
    if a2:
        text(L + 14, y + 30, a2, size=15, color=MUTED)
    text(L + 14 + cw[0], y, b, size=17, weight="bold", color=RED)
    text(L + 14 + cw[0] + cw[1], y, "\n".join(lines), size=16, color=SUB)
    y += rh
y += 40

# ---------------- 4 关键发现
y = section(y, "5", "关键发现", ORANGE)
finds = [
    ("瓶颈在功耗墙，不在 CPU", "满载 99% 以上的时间顶着 250W 功耗墙，SM 时钟中位 1350MHz（上限 1695）"),
    ("27B 不要跨卡 TP", "PCIe Gen2 且无 P2P，TP=2 比单卡慢 6~9 倍；只用单卡独立副本"),
    ("投机解码不改变输出分布", "贪心和拒绝采样下理论无损；实测分歧只出现在 bf16 分辨率内的近平局"),
    ("草稿词表要看语种", "上游 40k 草稿词表按丹麦语、英语、代码统计，中文基本不在表里；\n改全词表起草后，中文单流 68→127 tok/s"),
    ("先定噪声带再下结论", "同配置复测：通用负载 ±2%，开发场景单会话 ±0.5%，并发聚合 ±5%"),
]
for a, b in finds:
    text(L + 10, y, "• " + a, size=19, weight="bold")
    text(L + 38, y + 36, "\n".join(b.split("\n") if "\n" in b else wrap(b, 42)), size=16, color=SUB)
    y += 36 + 30 * (b.count("\n") + 1 if "\n" in b else len(wrap(b, 42))) + 22
y += 30

# ---------------- 5 最终配置
y = section(y, "6", "最终配置一览", BLUE)
box(L, y, R - L, 400, "#E3F2FD")
cfg = [
    ("推理引擎", "vLLM 0.30（HyperQwen PR#189 自建镜像）"),
    ("主模型", "Qwen3.8-27B W4A16 AutoRound，输出头 int8"),
    ("投机解码", "MTP k=4，全词表起草，起草头 int4（MTP_DRAFT_HEAD4=1）"),
    ("KV 缓存", "bf16"),
    ("部署", "每卡一个副本，共 7 副本；滚动脚本逐个重建并自检"),
    ("网关", "litellm simple-shuffle + session_affinity（TTL 3600s）"),
    ("思考", "网关默认关思考（enable_thinking: false），客户端可覆盖"),
]
for i, (k, v) in enumerate(cfg):
    text(L + 30, y + 26 + i * 52, k, size=18, weight="bold", color=BLUE)
    text(L + 180, y + 26 + i * 52, v, size=18)
y += 440
text(W / 2, y, "测试脚本、补丁与原始报告见仓库 170hx-qwen27b-inference-tuning", size=15, color=MUTED, ha="center")
text(W / 2, y + 30, "测试期间曾因外部供电中断，受影响的轮次已整体重测", size=15, color=MUTED, ha="center")
END = y + 70
print("content_end_px", END)
fig.savefig(OUT, dpi=100, facecolor="white")
