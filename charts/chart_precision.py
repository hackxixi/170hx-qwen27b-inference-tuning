import matplotlib; matplotlib.use("Agg")
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LogNorm
import numpy as np
for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc","/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]: fm.fontManager.addfont(f)
plt.rcParams["font.family"]=["Noto Sans CJK JP"]
names=["HyperQwen 现役","170hx-fullstack 原方案","170hx-fullstack 推理栈\n+ HyperQwen 模型"]
full=["HyperQwen 现役 · vLLM + MTP","170hx-fullstack 原方案 · SGLang + DFlash2 + EfficientThink","170hx-fullstack 推理栈 + HyperQwen 模型"]
colors=["#2E7D32","#E65100","#1565C0"]
fig=plt.figure(figsize=(10.8,24),dpi=100,facecolor="white")
fig.text(0.5,0.975,"Qwen3.8-27B 推理方案精度对比",ha="center",fontsize=34,weight="bold")
fig.text(0.5,0.957,"理论推算：group=128 数值模拟 + 各组件误差叠加 · 2026-09-24",ha="center",fontsize=17,color="#555")
for i,(n,c) in enumerate(zip(full,colors)):
    y=0.93-i*0.028
    fig.patches.append(plt.Rectangle((0.06,y-0.006),0.035,0.014,transform=fig.transFigure,color=c))
    fig.text(0.11,y+0.001,n,fontsize=17,va="center")

# Panel 1: total main-weight noise relative (lower better)
ax=fig.add_axes([0.30,0.70,0.64,0.10])
lo=[1.00,0.78,1.007]; hi=[1.00,0.86,1.007]; mid=[(a+b)/2 for a,b in zip(lo,hi)]
y=np.arange(3)
ax.barh(y,mid,color=colors,height=0.6,xerr=[[m-l for m,l in zip(mid,lo)],[h-m for m,h in zip(mid,hi)]],capsize=8,error_kw=dict(lw=2))
labs=["1.00（基准）","0.78–0.86","≈1.007"]
for yy,h,l in zip(y,hi,labs): ax.text(h+0.01,yy,l,va="center",fontsize=16,weight="bold")
ax.set_yticks(y); ax.set_yticklabels(names,fontsize=15); ax.invert_yaxis(); ax.set_xlim(0,1.3)
ax.set_title("③ 模型整体量化噪声（RMS，越低越接近原模型）",fontsize=21,loc="left",weight="bold",pad=12,x=-0.38)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.set_xticks([])
fig.text(0.06,0.672,"差距约 1.3–2.2 dB：原方案用非对称 int4、另把约 10% 敏感参数留 int8；\n现役的 AutoRound 校准通常优于 AWQ，会抵消一部分。",fontsize=14.5,color="#444",linespacing=1.5)

# Panel 2: heatmap per component
comps=["主体权重","输出头\nlm_head","词嵌入\nembed","KV 缓存","投机解码"]
val=np.array([[13.6,13.6,0.8,0.17,0.001],[11.7,0.17,0.17,2.65,0.001],[13.6,13.6,0.8,2.65,0.001]])
txt=[["int4 对称\n11–16%","int4\n11–16%","int8\n0.7–1%","bf16\n0.17%","无损\n0"],
     ["int4 非对称\n10–13%","bf16\n0.17%","bf16\n0.17%","fp8\n2.7%*","无损\n0"],
     ["int4 对称\n11–16%","int4 数值\n11–16%","int8 数值\n0.7–1%","fp8\n2.7%*","无损\n0"]]
ax=fig.add_axes([0.30,0.43,0.64,0.18])
ax.imshow(val,cmap="RdYlGn_r",norm=LogNorm(0.05,20),aspect="auto")
for i in range(3):
    for j in range(5):
        ax.text(j,i,txt[i][j],ha="center",va="center",fontsize=14,weight="bold",color="black",linespacing=1.3)
ax.set_xticks(range(5)); ax.set_xticklabels(comps,fontsize=15); ax.xaxis.tick_top()
ax.set_yticks(range(3)); ax.set_yticklabels(names,fontsize=15)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("④ 各组件相对误差（格式 + 误差 RMS，越绿越好）",fontsize=21,loc="left",weight="bold",pad=62,x=-0.38)
fig.text(0.06,0.365,"* fp8 KV 只影响 64 层中 16 层全注意力层，折算总噪声仅 +0.7%，可忽略。\n• 现役的主要短板是 int4 输出头：只可能翻转 top1 概率 <~60% 的“分岔 token”，\n  数学/代码等高置信输出几乎不受影响。\n• 投机解码（MTP / DFlash2）在贪心与拒绝采样下严格不改变输出，只影响速度。",fontsize=14.5,color="#444",linespacing=1.6)

# Panel 3: finetune effect
ax=fig.add_axes([0.30,0.13,0.64,0.17])
bm=["GPQA\n(n=198)","MMLU\n(n=500)","LiveCodeBench\n(n=100)"]
off=[82.8,90.2,69.0]; eff=[86.4,88.4,74.0]; se=[3.6,1.9,6.4]
d=[e-o for e,o in zip(eff,off)]
y=np.arange(3)
ax.barh(y,d,color=["#E65100" if x>0 else "#8D6E63" for x in d],height=0.5,zorder=2)
ax.errorbar(d,y,xerr=se,fmt="none",ecolor="black",capsize=8,lw=2,zorder=3)
for yy,x,o,e in zip(y,d,off,eff):
    ax.text(12.5,yy,f"{x:+.1f}\n{o}→{e}",va="center",fontsize=14,weight="bold")
ax.axvline(0,color="#333",lw=1.2)
ax.set_yticks(y); ax.set_yticklabels(bm,fontsize=15); ax.invert_yaxis(); ax.set_xlim(-10,18); ax.set_xticks([-10,-5,0,5,10])
ax.set_xlabel("EfficientThink 微调 − 官方权重（分），黑线 = ±1 个标准误",fontsize=14)
ax.set_title("⑤ 原方案的模型微调影响（该项目自测，开思考，FP8）",fontsize=21,loc="left",weight="bold",pad=12,x=-0.38)
for s in ["top","right"]: ax.spines[s].set_visible(False)
fig.text(0.06,0.075,"三项差异都在约 1σ 以内，统计上不显著；平均推理长度 −9%~+3%。\n关思考场景（本次测速条件）无数据。",fontsize=14.5,color="#444",linespacing=1.5)

fig.text(0.06,0.012,"结论：理论保真度 原方案 ≳ HyperQwen 现役 ≈ 推理栈+HyperQwen，差距约 1–2 dB，\n实际任务中难以区分。现役若要补精度，最划算的是把输出头换回 int8。",fontsize=16,color="#111",weight="bold",linespacing=1.6)
fig.savefig("precision-compare-mobile.png",dpi=100,facecolor="white")

# ---- 插入「① 模型结构」「② 量化方案明细」两块，拼到图例下方
from PIL import Image
W=10.8
bf=plt.figure(figsize=(W,17.2),dpi=100,facecolor="white")
def T(x,y,t,**k): bf.text(x,y,t,va="top",**k)
T(0.05,0.985,"① 模型结构（三套方案同为 Qwen3.8-27B 系）",fontsize=21,weight="bold")
arch=[("规模","约 27B 参数，dense（非 MoE），多模态（Qwen3_5ForConditionalGeneration）"),
("层数","64 层 = 48 层线性注意力（Gated DeltaNet）+ 16 层全注意力（每 4 层 1 层）"),
("宽度","隐藏维 5120；FFN 17408（SwiGLU）"),
("全注意力","24 个 Q 头 / 4 个 KV 头（GQA），head_dim 256，RoPE 只旋转 25% 维度"),
("线性注意力","16 个 K 头 / 48 个 V 头，维度 128，短卷积核 4；状态大小固定，不随上下文增长"),
("KV 缓存","只存在于 16 层全注意力 → 长上下文显存与 KV 量化误差都只涉及 1/4 的层"),
("词表 / 输出头","248,320 词；输出头与词嵌入不共享，各 1.27B 参数"),
("附属模块","MTP 头 1 层（投机起草）；视觉塔 27 层 ViT（1152 维）；上下文 262,144")]
y=0.945
for k,v in arch:
    T(0.06,y,k,fontsize=14,weight="bold",color="#1565C0"); T(0.22,y,v,fontsize=14); y-=0.03
y-=0.015
T(0.05,y,"② 量化方案明细",fontsize=21,weight="bold"); y-=0.04
cols=[0.05,0.215,0.475,0.735]; heads=["组件","HyperQwen 现役","170hx-fullstack 原方案","推理栈 + HyperQwen 模型"]
hc=["#333","#2E7D32","#E65100","#1565C0"]
for x,h,c in zip(cols,heads,hc): T(x,y,h,fontsize=14,weight="bold",color=c)
y-=0.028
rows=[("基座权重","官方 Qwen3.8-27B","EfficientThink 微调版\n（SFT + SimPO）","官方 Qwen3.8-27B"),
("量化格式","compressed-tensors W4A16\nAutoRound 校准","compressed-tensors W4A16\nAWQ 校准","同现役"),
("主体线性层","int4 对称 g128（全部）","int4 非对称 g128（367 个模块）\n+ int8 对称（33 个敏感模块）","同现役"),
("输出头","int4 g128\n（2026-09-25 起改 int8）","bf16","反量化成 bf16\n（数值同 int4）"),
("词嵌入","int8 g128","bf16","反量化成 bf16\n（数值同 int8）"),
("MTP 头","int4","bf16（未使用）","未使用"),
("保持 bf16","线性注意力 in_proj_a/b、视觉塔","线性注意力 a/b 与 norm、视觉塔","同现役"),
("KV 缓存","bf16","fp8_e4m3","fp8_e4m3"),
("线性注意力状态","bf16","bf16","bf16"),
("投机草稿","MTP 头，k=4","DFlash2 草稿模型\n（bf16，约 3.6GB），块 8","同原方案"),
("推理引擎","vLLM 0.28（Marlin 内核）","SGLang 0.5.19","SGLang 0.5.19")]
for i,r in enumerate(rows):
    n=max(t.count("\n")+1 for t in r); h=0.016*n+0.009
    if i%2==0: bf.patches.append(plt.Rectangle((0.04,y-h+0.006),0.93,h,transform=bf.transFigure,color="#F4F6F8",zorder=0))
    for x,t,j in zip(cols,r,range(4)): T(x,y,t,fontsize=12.5,weight="bold" if j==0 else "normal",linespacing=1.35)
    y-=h
T(0.05,y-0.006,"注：g128 = 每 128 个权重共用一个缩放系数。输出头改 int8 后，现役相对误差从 11~16% 降到 0.65~0.95%。",fontsize=12,color="#555")
bf.savefig("_block.png",dpi=100,facecolor="white"); plt.close(bf)
a=Image.open("precision-compare-mobile.png"); b=Image.open("_block.png"); cut=350
b=b.crop((0,0,b.width,int(b.height*(1-(y-0.03))) ))
out=Image.new("RGB",(a.width,a.height+b.height),"white")
out.paste(a.crop((0,0,a.width,cut)),(0,0)); out.paste(b,(0,cut)); out.paste(a.crop((0,cut,a.width,a.height)),(0,cut+b.height))
out.save("precision-compare-mobile.png"); os.remove("_block.png")
