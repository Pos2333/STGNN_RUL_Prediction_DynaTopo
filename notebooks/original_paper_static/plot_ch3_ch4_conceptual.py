# ============================================================
# plot_ch3_ch4_conceptual.py
# 第3~4章 概念示意图（纯 matplotlib 绘制，无数据依赖）
# 风格: seaborn Nature 期刊学术风格 + 中文标注
# ============================================================
import os, sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch, Circle
import seaborn as sns

# === Seaborn 学术风格 ===
sns.set_theme(style='white', context='paper', font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Nature 风格配色 (10色)
C = {'red':'#D62728','blue':'#1F77B4','green':'#2CA02C','orange':'#FF7F0E',
     'purple':'#9467BD','teal':'#17BECF','dark':'#2C3E50','gray':'#7F8C8D','yellow':'#BCBD22'}

print("="*60)
print("  开始绘制第3~4章概念图 (Nature 学术风格)...")
print("="*60)


# ======================== 图3: MSTCN 多尺度感受野 ========================
def draw_mstcn():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(-2, 33); ax.set_ylim(-1, 4.5)
    W = 30; kernels = [3,5,7]
    k_colors = [C['blue'], C['orange'], C['green']]
    k_names = ['k=3  短期局部模式\n     (工况切换等快速扰动)',
               'k=5  中期退化趋势\n     (部件性能渐进衰退)',
               'k=7  长期退化规律\n     (健康→失效宏观轨迹)']

    # 时间轴
    for t in range(W):
        ax.axvline(x=t, color='#bdc3c7', linewidth=0.4, alpha=0.4)
        if t%5==0: ax.text(t, -0.5, f't={t}', ha='center', fontsize=8, color=C['dark'])
    ax.plot([-0.5,29.5], [0,0], 'k-', linewidth=2)
    ax.text(15, -0.85, '时间轴 (运行周期)', ha='center', fontsize=12, fontweight='bold', color=C['dark'])

    # 当前时刻
    now_x = 24
    ax.axvline(x=now_x, color=C['red'], linewidth=2.5, linestyle='--', alpha=0.75)
    ax.text(now_x, 4.4, '当前\n时刻', ha='center', fontsize=9, color=C['red'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDE8E8', edgecolor=C['red'], alpha=0.9))

    for idx, (k, color, name) in enumerate(zip(kernels, k_colors, k_names)):
        y_base = 1.2 + idx * 1.0
        pad = k//2; start, end = now_x-pad, now_x+pad
        rect = FancyBboxPatch((start, y_base-0.25), k, 0.5, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor=color, alpha=0.22, linewidth=2)
        ax.add_patch(rect)
        ax.annotate('', xy=(end,y_base), xytext=(start,y_base),
                    arrowprops=dict(arrowstyle='<->', color=color, lw=3))
        ax.plot(now_x, y_base, 'o', color=color, markersize=10, zorder=5)
        ax.plot(now_x, y_base, 'o', color='white', markersize=4, zorder=6)
        ax.text(-1.8, y_base, name, ha='right', va='center', fontsize=8.5, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9))
        ax.text(start-0.3, y_base+0.35, f'感受野={k}', fontsize=8, color=color, fontweight='bold', ha='right')

    ax.text(15, 4.0, 'MSTCN 多尺度感受野对比', ha='center', fontsize=14, fontweight='bold', color=C['dark'])
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'ch3_mstcn_receptive_field.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("图3 已保存: ch3_mstcn_receptive_field.png")

draw_mstcn()


# ======================== 图4: GAT 消息传递机制 ========================
def draw_gat():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    coords = {'v0':(0,0),'v1':(-1.2,1.0),'v2':(1.0,1.0),'v3':(-1.0,-1.0),'v4':(1.0,-0.8)}
    node_labels = {'v0':'v0  T30\n高压压气机\n出口温度','v1':'v1  T24\n低压压气机\n出口温度',
                   'v2':'v2  P30\n高压压气机\n出口压力','v3':'v3  Nc\n核心转速',
                   'v4':'v4  NRc\n修正转速'}
    node_colors = {'v0':C['red'],'v1':C['orange'],'v2':C['blue'],'v3':C['green'],'v4':C['green']}
    attn = {('v1','v0'):0.45,('v2','v0'):0.35,('v3','v0'):0.12,('v4','v0'):0.08}

    for ax, title, show_arrows in [(ax1,'聚合前 (仅网络拓扑)',False),(ax2,'聚合后 (注意力加权)',True)]:
        ax.set_xlim(-2,2); ax.set_ylim(-1.8,1.8); ax.set_aspect('equal')
        for nei in ['v1','v2','v3','v4']:
            nx_,ny_=coords[nei]; cx,cy=coords['v0']
            if show_arrows:
                w=attn[(nei,'v0')]
                ax.annotate('', xy=(cx,cy), xytext=(nx_,ny_),
                           arrowprops=dict(arrowstyle='->',color=C['dark'],lw=1.5+w*15,
                                          alpha=0.4+w,connectionstyle='arc3,rad=0.1'))
                mx,my=(nx_+cx)/2,(ny_+cy)/2
                ax.text(mx+0.15,my+0.12,f'a={w:.2f}',fontsize=8,color=C['red'],fontweight='bold')
            else:
                ax.plot([nx_,cx],[ny_,cy],color=C['gray'],linewidth=1.5,alpha=0.45,linestyle='--')
        for v,(x,y) in coords.items():
            sz,ew = (2200,3) if v=='v0' else (1400,1.5)
            r = 0.38 if v=='v0' else 0.30
            ax.add_patch(Circle((x,y),r,facecolor=node_colors[v],edgecolor='white',linewidth=ew,alpha=0.9,zorder=5))
            ax.text(x,y,node_labels[v],ha='center',va='center',fontsize=6.5,fontweight='bold',color='white',zorder=6)
        ax.set_title(title, fontsize=13, fontweight='bold', color=C['dark'], pad=10); ax.axis('off')

    fig.suptitle('GAT 图注意力消息传递机制', fontsize=15, fontweight='bold', y=1.03)
    fig.text(0.5, -0.01, '箭头粗细正比于注意力权重', ha='center', fontsize=9, fontstyle='italic', color=C['gray'])
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'ch3_gat_message_passing.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("图4 已保存: ch3_gat_message_passing.png")

draw_gat()


# ======================== 图5: STGNN 数据流图 ========================
def draw_stgnn():
    modules = [
        ('输入样本\n[B, 30, 17]',         C['dark'],   3.5, 8.5, 2.4, 1.2),
        ('拆分\n操作参数 / 传感器',        C['gray'],   3.5, 7.2, 1.2, 0.6),
        ('操作参数\n[B, 3, W] → [B, 16]', C['teal'],   1.5, 4.8, 2.4, 1.1),
        ('传感器数据\n[B, W, 14]',          C['gray'],   5.5, 5.8, 2.0, 0.7),
        ('MSTCN\n多尺度时序特征\n[B, 14, 128]', C['orange'], 4.0, 4.4, 2.5, 1.2),
        ('GAT\n图注意力空间建模\n[B, 64]',       C['purple'], 7.0, 4.4, 2.5, 1.2),
        ('特征拼接\nConcat [B, 208]',      C['blue'],   4.0, 2.8, 2.4, 1.0),
        ('全连接输出\nFC + ReLU [B, 1]',   C['green'],  4.0, 1.4, 2.2, 0.9),
        ('RUL 预测值\ny_hat in [0, 125]',       C['red'],    4.0, 0.2, 2.4, 0.8),
    ]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 9); ax.set_ylim(-0.8, 9.5); ax.axis('off')

    boxes = {}
    for name, color, cx, cy, w, h in modules:
        rect = FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor='white', linewidth=2.2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(cx, cy, name, ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')
        boxes[name] = (cx, cy, w, h)

    arrows = [
        ('输入样本\n[B, 30, 17]','拆分\n操作参数 / 传感器',''),
        ('拆分\n操作参数 / 传感器','操作参数\n[B, 3, W] → [B, 16]','permute\n(0,2,1)'),
        ('拆分\n操作参数 / 传感器','传感器数据\n[B, W, 14]',''),
        ('传感器数据\n[B, W, 14]','MSTCN\n多尺度时序特征\n[B, 14, 128]','permute\n(0,2,1)'),
        ('MSTCN\n多尺度时序特征\n[B, 14, 128]','GAT\n图注意力空间建模\n[B, 64]','邻接矩阵\nedge_index'),
        ('操作参数\n[B, 3, W] → [B, 16]','特征拼接\nConcat [B, 208]','池化'),
        ('GAT\n图注意力空间建模\n[B, 64]','特征拼接\nConcat [B, 208]','均值池化'),
        ('特征拼接\nConcat [B, 208]','全连接输出\nFC + ReLU [B, 1]','Dropout'),
        ('全连接输出\nFC + ReLU [B, 1]','RUL 预测值\ny_hat in [0, 125]',''),
    ]
    for src, dst, label in arrows:
        if src in boxes and dst in boxes:
            sx,sy,sw,sh = boxes[src]; dx,dy,dw,dh = boxes[dst]
            ax.annotate('', xy=(dx,dy+dh/2), xytext=(sx,sy-sh/2),
                       arrowprops=dict(arrowstyle='->',color=C['dark'],lw=2.2))
            if label:
                ax.text((sx+dx)/2,(sy-sh/2+dy+dh/2)/2,label,ha='center',va='center',fontsize=7,
                       color=C['red'],fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2',facecolor='white',edgecolor=C['red'],alpha=0.85))

    legend_items = [('输入',C['dark']),('时序特征',C['orange']),('空间特征',C['purple']),
                    ('特征融合',C['blue']),('输出',C['green'])]
    ax.legend(handles=[mpatches.Patch(color=c,label=n,alpha=0.9) for n,c in legend_items],
             loc='upper right',fontsize=10,title='模块类型',title_fontsize=11,framealpha=0.9)
    ax.text(4.5,9.3,'STGNN 时空图神经网络架构与数据流',ha='center',fontsize=16,fontweight='bold',color=C['dark'])
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,'ch3_stgnn_dataflow.png'),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print("图5 已保存: ch3_stgnn_dataflow.png")

draw_stgnn()


# ======================== 图6: LMMD 子域对齐 vs 全局 MMD ========================
def draw_lmmd():
    """纯概念示意图：用椭圆 + 标注，避免看起来像真实数据散点"""
    np.random.seed(42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # 5个子域的颜色（统一调色板）
    sub_colors = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#3498DB']
    sub_names = ['临近失效\nRUL∈[0,25]', '严重退化\nRUL∈[25,50]',
                 '中期退化\nRUL∈[50,75]', '轻度退化\nRUL∈[75,100]',
                 '健康阶段\nRUL∈[100,125]']

    # 源域5个子域中心 + 目标域偏移
    src_centers = [(2.0, 5.0), (2.5, 3.8), (3.0, 2.6), (3.5, 1.5), (4.0, 0.6)]
    tgt_shift = (2.5, 1.2)

    # ====== 左图：全局 MMD（概念椭圆表示分布云） ======
    ax1.set_xlim(-0.5, 10.5); ax1.set_ylim(-0.5, 7)
    ax1.set_title('全局 MMD 对齐\nGlobal MMD — 退化阶段混淆', fontsize=14,
                  fontweight='bold', color=C['red'], pad=14)

    # 画源域和目标域的分布椭圆（半透明大椭圆覆盖整个域）
    from matplotlib.patches import Ellipse
    # 源域整体椭圆
    src_all = Ellipse((3.0, 3.0), width=4.5, height=5.5, angle=15,
                      facecolor='#3498DB', edgecolor='#2980B9', alpha=0.12, linewidth=2, linestyle='--')
    ax1.add_patch(src_all)
    # 目标域整体椭圆
    tgt_all = Ellipse((5.5, 4.2), width=4.5, height=5.5, angle=15,
                      facecolor='#E74C3C', edgecolor='#C0392B', alpha=0.12, linewidth=2, linestyle='--')
    ax1.add_patch(tgt_all)

    # 画各子域椭圆
    for i, (color, (sx, sy)) in enumerate(zip(sub_colors, src_centers)):
        tx, ty = sx + tgt_shift[0], sy + tgt_shift[1]
        for (x, y), _domain, alpha in [((sx, sy), 'src', 0.35), ((tx, ty), 'tgt', 0.35)]:
            ell = Ellipse((x, y), width=1.2, height=0.7, angle=-20,
                         facecolor=color, edgecolor='white', alpha=alpha, linewidth=1.5)
            ax1.add_patch(ell)
        # 子域内标签
        ax1.text(sx, sy + 0.55, sub_names[i], ha='center', fontsize=6.5,
                fontweight='bold', color=color)

    # 错误对齐箭头（健康阶段 → 临近失效）
    wrong_arrows = [((2.0, 5.0), (6.5, 1.8)), ((3.5, 1.5), (5.0, 5.5))]
    for src, dst in wrong_arrows:
        ax1.annotate('', xy=dst, xytext=src,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.8, linestyle='--', alpha=0.5,
                                   connectionstyle='arc3,rad=0.3'))

    ax1.text(5.0, 6.5, '将不同退化阶段的\n样本错误拉近', ha='center', fontsize=10,
            color=C['red'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDE8E8', edgecolor=C['red'], alpha=0.9))
    ax1.set_xticks([]); ax1.set_yticks([])

    # ====== 右图：LMMD 子域对齐（同一颜色子域内对齐） ======
    ax2.set_xlim(-0.5, 10.5); ax2.set_ylim(-0.5, 7)
    ax2.set_title('LMMD 子域对齐\nLMMD — 退化阶段匹配', fontsize=14,
                  fontweight='bold', color=C['green'], pad=14)

    for i, (color, (sx, sy)) in enumerate(zip(sub_colors, src_centers)):
        tx, ty = sx + tgt_shift[0], sy + tgt_shift[1]
        # 源域和目标域的子域椭圆
        for (x, y), _domain, alpha in [((sx, sy), 'src', 0.4), ((tx, ty), 'tgt', 0.4)]:
            ell = Ellipse((x, y), width=1.2, height=0.7, angle=-20,
                         facecolor=color, edgecolor='white', alpha=alpha, linewidth=1.5)
            ax2.add_patch(ell)
        # 子域内对齐箭头（同色）
        ax2.annotate('', xy=((sx+tx)/2, (sy+ty)/2), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color=color, lw=3, alpha=0.7))
        # 子域名标签
        ax2.text((sx+tx)/2 + 0.25, (sy+ty)/2 + 0.4, sub_names[i], ha='center',
                fontsize=6.5, fontweight='bold', color=color)

    ax2.text(5.0, 6.5, '仅在相同退化阶段内\n进行局部分布匹配', ha='center', fontsize=10,
            color=C['green'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F8F5', edgecolor=C['green'], alpha=0.9))
    ax2.set_xticks([]); ax2.set_yticks([])

    # 全局图例
    fig.legend(handles=[
        mpatches.Patch(color='#3498DB', alpha=0.3, label='源域分布 (FD001)'),
        mpatches.Patch(color='#E74C3C', alpha=0.3, label='目标域分布 (FD002~4)'),
    ], loc='lower center', ncol=2, fontsize=10.5, framealpha=0.9)

    fig.suptitle('分布对齐策略对比: 全局 MMD vs LMMD 子域对齐（概念示意图）',
                 fontsize=15, fontweight='bold', y=1.03)
    fig.text(0.5, -0.01, '注: 椭圆表示特征分布云, 颜色表示退化阶段, 箭头表示对齐方向',
             ha='center', fontsize=9, fontstyle='italic', color=C['gray'])

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'ch4_lmmd_vs_mmd.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("图6 已保存: ch4_lmmd_vs_mmd.png (概念椭圆版)")

draw_lmmd()

print("\n[完成] 第3~4章概念图全部完成!")
