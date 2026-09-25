import matplotlib; matplotlib.use("Agg")
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc","/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]: fm.fontManager.addfont(f)
plt.rcParams["font.family"]=["Noto Sans CJK JP"]
names=["HyperQwen 现役（int4 输出头）","HyperQwen + int8 输出头","170hx-fullstack 原方案","170hx-fullstack 推理栈 + HyperQwen 模型"]
colors=["#2E7D32","#81C784","#E65100","#1565C0"]
fig=plt.figure(figsize=(10.8,24),dpi=100,facecolor="white")
fig.text(0.5,0.978,"开发场景速度对比 + int8 输出头实测",ha="center",fontsize=32,weight="bold")
fig.text(0.5,0.962,"同一节点 · CMP 170HX 64GB · 23K 前缀 + 8 条多轮 agent 会话 · 2026-09-25",ha="center",fontsize=16,color="#555")
for i,(n,c) in enumerate(zip(names,colors)):
    y=0.94-i*0.022
    fig.patches.append(plt.Rectangle((0.06,y-0.005),0.035,0.012,transform=fig.transFigure,color=c))
    fig.text(0.11,y+0.001,n,fontsize=17,va="center")
def panel(bot,h,title,cats,vals,hib,fmt="{:g}"):
    ax=fig.add_axes([0.2,bot,0.74,h])
    y=np.arange(len(cats)); n=len(vals); bh=0.8/n
    best=[(max if hib else min)(vals[m][j] for m in range(n)) for j in range(len(cats))]
    for i in range(n):
        yy=y+(i-(n-1)/2)*bh
        ax.barh(yy,[vals[i][j] for j in range(len(cats))],height=bh*0.9,color=colors[i])
        for j,v in enumerate(vals[i]):
            b=v==best[j]; ax.text(v,yy[j],"  "+fmt.format(v)+(" ★" if b else ""),va="center",fontsize=14,weight="bold" if b else "normal")
    ax.set_yticks(y); ax.set_yticklabels(cats,fontsize=17); ax.invert_yaxis()
    ax.set_xlim(0,max(max(r) for r in vals)*1.28); ax.set_xticks([])
    ax.set_title(title,fontsize=21,loc="left",weight="bold",pad=8,x=-0.2)
    for s in ["top","right","bottom"]: ax.spines[s].set_visible(False)
panel(0.655,0.19,"① 生成速度（tok/s）",["单会话 decode","4 并发总吞吐","8 并发总吞吐"],
      [[120,132,138],[120,125,136],[114,128,105],[128,124,117]],True)
panel(0.555,0.055,"② 8 并发每条会话耗时（秒，越低越好）",["会话耗时"],[[55.3],[56.0],[77.4],[71.0]],False)
panel(0.39,0.12,"③ 后续轮首字延迟（秒，越低越好）",["单会话","8 并发"],[[0.73,1.21],[0.74,1.34],[0.51,3.43],[0.51,2.05]],False,"{:.2f}")
# int8 head change panel
ax=fig.add_axes([0.2,0.19,0.74,0.15])
cats=["单流 英文","单流 中文","8 并发 英文","8 并发 中文","4k 长输入","32k 长输入"]
d=[-4.2,-0.1,-2.0,-4.4,-7.2,1.3]; y=np.arange(len(cats))
ax.barh(y,d,color=["#C62828" if v<0 else "#2E7D32" for v in d],height=0.6)
for yy,v in zip(y,d): ax.text(v+(-0.3 if v<0 else 0.3),yy,f"{v:+.1f}%",va="center",ha="right" if v<0 else "left",fontsize=15,weight="bold")
ax.axvline(0,color="#333",lw=1); ax.set_xlim(-10,4); ax.set_xticks([])
ax.set_yticks(y); ax.set_yticklabels(cats,fontsize=15); ax.invert_yaxis()
for s in ["top","right","bottom"]: ax.spines[s].set_visible(False)
ax.set_title("④ int8 输出头相对现役的速度变化（通用负载）",fontsize=21,loc="left",weight="bold",pad=8,x=-0.2)
fig.text(0.06,0.125,"精度：int8 输出头与现役的首选 token 一致率 97.6%（测量噪声底 99.45%），\n约每 55 个 token 有 1 处选词不同，都在近平局位置。",fontsize=15,color="#333",linespacing=1.6)
fig.text(0.06,0.08,"工具调用：四套函数名 100% 合法；JSON 参数合法率 96–100%，\n不合法的都是输出撞到 max_tokens 被截断。",fontsize=15,color="#333",linespacing=1.6)
fig.text(0.06,0.012,"★ 该项最优。本负载下：有并发时 HyperQwen 现役吞吐更高（8 并发高 18–32%）；\n170hx-fullstack 推理栈在单会话快约 7%。\nint8 输出头速度代价 0–5%（长输入 7%），已于 2026-09-25 上线。",fontsize=16,color="#111",weight="bold",linespacing=1.6)
fig.savefig("dev-compare-mobile.png",dpi=100,facecolor="white")
