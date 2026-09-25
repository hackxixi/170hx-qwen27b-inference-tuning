import matplotlib; matplotlib.use("Agg")
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc","/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]: fm.fontManager.addfont(f)
print(sorted({x.name for x in fm.fontManager.ttflist if "CJK" in x.name}))
plt.rcParams["font.family"]=["Noto Sans CJK JP"]
names=["HyperQwen 现役\nvLLM + MTP","170hx-fullstack 原方案\nSGLang + DFlash2 + EfficientThink","170hx-fullstack 推理栈 + HyperQwen 模型"]
colors=["#2E7D32","#E65100","#1565C0"]
panels=[
 ("单流生成速度（每路 tok/s）",["英文","中文"],[[159,126],[177,101],[188,111]],"tok/s",True),
 ("4 并发总吞吐（tok/s）",["英文","中文"],[[383,330],[290,258],[333,274]],"tok/s",True),
 ("8 并发总吞吐（tok/s）",["英文","中文"],[[631,541],[507,365],[509,385]],"tok/s",True),
 ("长输入单流生成速度（tok/s）",["4k 输入","32k 输入"],[[159,121],[103,90],[137,93]],"tok/s",True),
 ("长输入首字延迟（秒，越低越好）",["4k 输入","32k 输入"],[[2.0,16.1],[2.6,18.8],[2.0,18.4]],"s",False),
]
fig=plt.figure(figsize=(10.8,24),dpi=100,facecolor="white")
fig.text(0.5,0.975,"Qwen3.8-27B 推理方案速度对比",ha="center",fontsize=34,weight="bold")
fig.text(0.5,0.957,"同一张 CMP 170HX 64GB · 贪心解码 · 关思考 · 2 轮中位数 · 2026-09-24",ha="center",fontsize=17,color="#555")
# legend
for i,(n,c) in enumerate(zip(names,colors)):
    y=0.93-i*0.028
    fig.patches.append(plt.Rectangle((0.08,y-0.006),0.035,0.014,transform=fig.transFigure,color=c))
    fig.text(0.13,y+0.001,n.replace("\n"," · "),fontsize=17,va="center")
top=0.835; h=0.118; gap=0.038
for k,(title,cats,vals,unit,hib) in enumerate(panels):
    ax=fig.add_axes([0.15,top-k*(h+gap)-h,0.79,h])
    ypos=np.arange(len(cats))
    bh=0.26
    for i in range(3):
        v=[vals[i][j] for j in range(len(cats))]
        yy=ypos+(i-1)*bh
        ax.barh(yy,v,height=bh*0.9,color=colors[i])
        best=[ (max if hib else min)(vals[m][j] for m in range(3)) for j in range(len(cats))]
        for y,x,j in zip(yy,v,range(len(cats))):
            lab=(f"{x:.1f}" if unit=="s" else f"{x:g}")+(" ★" if x==best[j] else "")
            ax.text(x,y,"  "+lab,va="center",fontsize=16,weight="bold" if x==best[j] else "normal")
    ax.set_yticks(ypos); ax.set_yticklabels(cats,fontsize=18); ax.invert_yaxis()
    mx=max(max(r) for r in vals); ax.set_xlim(0,mx*1.25)
    ax.set_title(title,fontsize=22,loc="left",weight="bold",pad=10)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.tick_params(axis="x",labelsize=13,colors="#777")
fig.text(0.06,0.012,"★ 该项最优。本负载下：HyperQwen 现役在中文、并发、长上下文吞吐更高；\n170hx-fullstack 英文单流更高（DFlash2 草稿中文接受率约 23%）。\n精度理论差距约 1–2 dB，无实质差异。",fontsize=16,color="#333",va="bottom",linespacing=1.6)
fig.savefig("speed-compare-mobile.png",dpi=100,facecolor="white")
