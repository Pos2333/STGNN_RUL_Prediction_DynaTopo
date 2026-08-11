# ============================================================
# plot_ch4_ch5_model_results.py
# 第4~5章 模型结果可视化 —— 3子图 t-SNE & 预测散点图
# 风格: seaborn Nature 期刊学术风格 + 中文标注
# ============================================================
import os, sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
from sklearn.manifold import TSNE

sns.set_theme(style='whitegrid', context='paper', font_scale=1.1,
              rc={'axes.edgecolor':'0.15','grid.alpha':0.2,
                  'figure.facecolor':'white','axes.facecolor':'#fafafa'})
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
from configs.config import (NUM_FEATURES,NUM_SENSORS,BATCH_SIZE,RANDOM_SEED,
    MSTCN_NUM_CHANNELS,MSTCN_KERNEL_SIZES,MSTCN_DROPOUT,
    GAT_HIDDEN_DIM,GAT_HEADS,GAT_DROPOUT,
    TRANSFORMER_D_MODEL,TRANSFORMER_NHEAD,TRANSFORMER_NUM_LAYERS,TRANSFORMER_DROPOUT,FC_HIDDEN_DIM)
from core_models.stgnn_static import STGNN_Static
from core_models.base_models import BasicLSTM, GRUModel, TCNModel, CNN_LSTM_Model
from utils.metrics import compute_rmse, compute_nasa_score

np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {DEVICE}")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

USE_SAMPLING = True; SAMPLE_SIZE = 2000
print(f"采样开关: {USE_SAMPLING}, 采样量: {SAMPLE_SIZE}")

PROCESSED_DIR = os.path.join(PROJ_ROOT,'data','processed')
MODEL_DIR = os.path.join(PROJ_ROOT,'saved_models')

C = {'blue':'#1F77B4','red':'#D62728','green':'#2CA02C','orange':'#FF7F0E',
     'purple':'#9467BD','teal':'#17BECF','dark':'#2C3E50','gray':'#7F8C8D'}

def load_test_data(subset='FD001'):
    X=np.load(os.path.join(PROCESSED_DIR,f'{subset}_test.npz'))['X']
    y=np.load(os.path.join(PROCESSED_DIR,f'{subset}_test.npz'))['y']
    ei=torch.load(os.path.join(PROCESSED_DIR,f'{subset}_train_graph.pt'),weights_only=False)['edge_index']
    print(f"  {subset}: {len(X)} 样本, y∈[{y.min():.0f},{y.max():.0f}]")
    return X,y,ei

def predict_stgnn(model,X,edge_index,return_features=False):
    edge_index=edge_index.to(DEVICE)
    loader=DataLoader(TensorDataset(torch.tensor(X,dtype=torch.float32)),batch_size=BATCH_SIZE,shuffle=False)
    model.eval();preds,feats=[],[]
    with torch.no_grad():
        for (Xb,) in loader:
            Xb=Xb.to(DEVICE)
            if return_features: p,f=model(Xb,edge_index,return_feat=True);feats.append(f.cpu().numpy())
            else: p=model(Xb,edge_index,return_feat=False)
            preds.append(p.cpu().numpy())
    if return_features:return np.concatenate(preds),np.concatenate(feats)
    return np.concatenate(preds)

def predict_generic(model,X):
    """通用推理函数，适用于 LSTM/GRU/TCN/CNN-LSTM 等模型"""
    loader=DataLoader(TensorDataset(torch.tensor(X,dtype=torch.float32)),batch_size=BATCH_SIZE,shuffle=False)
    model.eval();preds=[]
    with torch.no_grad():
        for (Xb,) in loader:preds.append(model(Xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)

def build_stgnn_v2():
    return STGNN(num_sensors=NUM_SENSORS,num_op_settings=3,
        mstcn_channels=MSTCN_NUM_CHANNELS,mstcn_kernels=MSTCN_KERNEL_SIZES,mstcn_dropout=MSTCN_DROPOUT,
        gat_hidden=GAT_HIDDEN_DIM,gat_heads=GAT_HEADS,gat_dropout=GAT_DROPOUT,
        trans_d_model=TRANSFORMER_D_MODEL,trans_nhead=TRANSFORMER_NHEAD,trans_num_layers=TRANSFORMER_NUM_LAYERS,
        trans_dropout=TRANSFORMER_DROPOUT,use_mstcn=True,use_gat=True,use_transformer=False,fc_hidden=FC_HIDDEN_DIM).to(DEVICE)

print("\n--- 加载数据 ---")
X_fd001,y_fd001,edge_fd001=load_test_data('FD001')
X_fd002,y_fd002,edge_fd002=load_test_data('FD002')

# ================================================================
# 图7: 3子图 t-SNE — RUL颜色编码版
# ================================================================
print("\n"+"="*60)
print("  图7: t-SNE 特征可视化 (3子图, RUL颜色编码)")
print("="*60)

if USE_SAMPLING:
    si=np.random.choice(len(X_fd001),min(SAMPLE_SIZE//2,len(X_fd001)),replace=False)
    ti=np.random.choice(len(X_fd002),min(SAMPLE_SIZE//2,len(X_fd002)),replace=False)
    X_s,y_s=X_fd001[si],y_fd001[si];X_t,y_t=X_fd002[ti],y_fd002[ti]
else:X_s,y_s,X_t,y_t=X_fd001,y_fd001,X_fd002,y_fd002
print(f"采样: 源域={len(X_s)}, 目标域={len(X_t)}")

# 三个模型特征
print("加载无迁移基线模型...")
m_no=build_stgnn_v2()
m_no.load_state_dict(torch.load(os.path.join(MODEL_DIR,'stgnn_static_best_FD001.pt'),map_location=DEVICE,weights_only=False)['model_state_dict'])
_,fs_no=predict_stgnn(m_no,X_s,edge_fd001,return_features=True)
_,ft_no=predict_stgnn(m_no,X_t,edge_fd002,return_features=True)

print("加载全局 MMD 迁移模型...")
m_mmd=build_stgnn_v2()
m_mmd.load_state_dict(torch.load(os.path.join(MODEL_DIR,'transfer_v2_global_mmd_best_FD002.pt'),map_location=DEVICE,weights_only=False)['model_state_dict'])
_,fs_mmd=predict_stgnn(m_mmd,X_s,edge_fd001,return_features=True)
_,ft_mmd=predict_stgnn(m_mmd,X_t,edge_fd002,return_features=True)

print("加载 LMMD 迁移模型...")
m_lmmd=build_stgnn_v2()
m_lmmd.load_state_dict(torch.load(os.path.join(MODEL_DIR,'transfer_static_best_FD002.pt'),map_location=DEVICE,weights_only=False)['model_state_dict'])
_,fs_lmmd=predict_stgnn(m_lmmd,X_s,edge_fd001,return_features=True)
_,ft_lmmd=predict_stgnn(m_lmmd,X_t,edge_fd002,return_features=True)

# === 分离 t-SNE：左+右共用，中图独立 ===
print("运行 t-SNE 降维 (左+右: 无迁移+LMMD)...")
all_lr=np.concatenate([fs_no,ft_no,fs_lmmd,ft_lmmd],axis=0)
tsne_lr=TSNE(n_components=2,perplexity=30,random_state=RANDOM_SEED,max_iter=1000).fit_transform(all_lr)
n_s,n_t=len(fs_no),len(ft_no);n_sl,n_tl=len(fs_lmmd),len(ft_lmmd)
o=0;tsne_no_s,tsne_no_t=tsne_lr[o:o+n_s],tsne_lr[o:o+n_s+n_t][n_s:];o+=n_s+n_t
tsne_lmmd_s,tsne_lmmd_t=tsne_lr[o:o+n_sl],tsne_lr[o:o+n_sl+n_tl][n_sl:]

print("运行 t-SNE 降维 (中: 无迁移+全局MMD)...")
all_mid=np.concatenate([fs_no,ft_no,fs_mmd,ft_mmd],axis=0)
tsne_mid=TSNE(n_components=2,perplexity=30,random_state=RANDOM_SEED,max_iter=1000).fit_transform(all_mid)
n_sm,n_tm=len(fs_mmd),len(ft_mmd)
o2=0;tsne_mid_s,tsne_mid_t=tsne_mid[o2:o2+n_s],tsne_mid[o2:o2+n_s+n_t][n_s:]
o2+=n_s+n_t;tsne_mmd_s,tsne_mmd_t=tsne_mid[o2:o2+n_sm],tsne_mid[o2:o2+n_sm+n_tm][n_sm:]

# === 3子图绘制 ===
from matplotlib.patches import Ellipse

fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(25,7.8))
rul_norm=plt.Normalize(0,125);rul_cmap=plt.cm.viridis
y_t_flat=y_t.flatten();y_s_flat=y_s.flatten()

# RUL 子域划分
rul_bins=[(0,25),(25,50),(50,75),(75,100),(100,125)]
ell_colors=['#440154','#3B528B','#21918C','#5EC962','#FDE725']
ell_labels=['0-25','25-50','50-75','75-100','100-125']

# 工具函数：计算子域质心距平均值
def subdomain_centroid_dist(tsne_s, tsne_t, y_src, y_tgt):
    """返回 5个子域的源→目标质心距列表 和 平均值"""
    dists=[]
    for lo,hi in rul_bins:
        sm=(y_src>=lo)&(y_src<hi);tm=(y_tgt>=lo)&(y_tgt<hi)
        if sm.sum()>2 and tm.sum()>2:
            d=np.linalg.norm(tsne_s[sm].mean(axis=0)-tsne_t[tm].mean(axis=0))
            dists.append(d)
    return dists,np.mean(dists) if dists else 0

# 画质心连线的工具函数（距离标注在虚线中点上方）
def draw_centroid_link(ax, src_pts, tgt_pts, color_s, color_t, label=''):
    sc=src_pts.mean(axis=0);tc=tgt_pts.mean(axis=0)
    ax.scatter(*sc,c=color_s,s=250,marker='X',edgecolors='white',linewidth=2,zorder=10)
    ax.scatter(*tc,c=color_t,s=250,marker='X',edgecolors='white',linewidth=2,zorder=10)
    ax.annotate('',xy=tc,xytext=sc,arrowprops=dict(arrowstyle='->',color='gray',lw=2,linestyle='--'))
    mid=(sc+tc)/2;d=np.linalg.norm(sc-tc)
    ax.text(mid[0],mid[1]+2.5,f'{label}={d:.1f}',ha='center',fontsize=9,fontweight='bold',color=C['dark'],
            bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor='gray',alpha=0.85))
    return sc,tc,d

# 画子域椭圆的工具函数
def draw_subdomain_ellipses(ax, pts, y_vals):
    for (lo,hi),ec,el in zip(rul_bins,ell_colors,ell_labels):
        mask=(y_vals>=lo)&(y_vals<hi);p=pts[mask]
        if len(p)>5:
            try:
                mu=p.mean(axis=0);cov=np.cov(p.T)
                eigval,eigvec=np.linalg.eigh(cov)
                angle=np.degrees(np.arctan2(eigvec[1,0],eigvec[0,0]))
                w,h=2*np.sqrt(np.maximum(eigval,0.01))*1.5
                ax.add_patch(Ellipse(mu,w,h,angle=angle,facecolor=ec,edgecolor=ec,alpha=0.15,linewidth=1.5))
                ax.text(mu[0],mu[1],el,ha='center',va='center',fontsize=6,fontweight='bold',color=ec)
            except:pass

# ======================== 左: 对齐前 ========================
ax1.scatter(tsne_no_s[:,0],tsne_no_s[:,1],c=C['blue'],alpha=0.5,s=20,label='源域 (FD001)')
ax1.scatter(tsne_no_t[:,0],tsne_no_t[:,1],c=C['red'],alpha=0.5,s=20,marker='^',label='目标域 (FD002)')
dists_no,avg_no=subdomain_centroid_dist(tsne_no_s,tsne_no_t,y_s_flat,y_t_flat)
draw_centroid_link(ax1,tsne_no_s,tsne_no_t,C['blue'],C['red'],'子域质心距均值')
ax1.set_title('(a) 对齐前 (无迁移)\n源域与目标域明显分离',
             fontsize=12,fontweight='bold',color=C['red'],pad=10)
ax1.legend(fontsize=8,framealpha=0.85,loc='upper right')

# ======================== 中: 全局 MMD ========================
sc2=ax2.scatter(tsne_mmd_t[:,0],tsne_mmd_t[:,1],c=y_t_flat,cmap=rul_cmap,norm=rul_norm,alpha=0.55,s=20,marker='^',edgecolors='none')
ax2.scatter(tsne_mmd_s[:,0],tsne_mmd_s[:,1],c='#5D6D7E',alpha=0.4,s=16,label='源域 (FD001)')
# 子域椭圆 (MMD下应该重叠混杂)
draw_subdomain_ellipses(ax2,tsne_mmd_t,y_t_flat)
dists_mmd,avg_mmd=subdomain_centroid_dist(tsne_mmd_s,tsne_mmd_t,y_s_flat,y_t_flat)
draw_centroid_link(ax2,tsne_mmd_s,tsne_mmd_t,'#5D6D7E',C['orange'],'子域质心距均值')
ax2.set_title('(b) 全局 MMD 对齐后\n分布拉近但 RUL 阶段混杂',
             fontsize=12,fontweight='bold',color=C['orange'],pad=10)
cbar2=plt.colorbar(sc2,ax=ax2,shrink=0.75);cbar2.set_label('RUL',fontsize=9)

# ======================== 右: LMMD ========================
sc3=ax3.scatter(tsne_lmmd_t[:,0],tsne_lmmd_t[:,1],c=y_t_flat,cmap=rul_cmap,norm=rul_norm,alpha=0.55,s=20,marker='^',edgecolors='none')
ax3.scatter(tsne_lmmd_s[:,0],tsne_lmmd_s[:,1],c='#5D6D7E',alpha=0.4,s=16,label='源域 (FD001)')
# 子域椭圆 (LMMD下应该分层有序)
draw_subdomain_ellipses(ax3,tsne_lmmd_t,y_t_flat)
dists_lmmd,avg_lmmd=subdomain_centroid_dist(tsne_lmmd_s,tsne_lmmd_t,y_s_flat,y_t_flat)
draw_centroid_link(ax3,tsne_lmmd_s,tsne_lmmd_t,'#5D6D7E',C['green'],'子域质心距均值')
ax3.set_title('(c) LMMD 子域对齐后\n分布对齐且 RUL 阶段保序',
             fontsize=12,fontweight='bold',color=C['green'],pad=10)
cbar3=plt.colorbar(sc3,ax=ax3,shrink=0.75);cbar3.set_label('RUL',fontsize=9)

for ax in[ax1,ax2,ax3]:ax.set_xlabel('t-SNE 维度 1',fontsize=10)
ax1.set_ylabel('t-SNE 维度 2',fontsize=10)

# 底部汇总
summary=(f'子域质心距均值:  无迁移={avg_no:.1f}  |  全局MMD={avg_mmd:.1f}  |  LMMD={avg_lmmd:.1f}'
         f'    (子域内对齐: MMD={np.std(dists_mmd) if dists_mmd else 0:.1f}±{avg_mmd:.1f}, '
         f'LMMD={np.std(dists_lmmd) if dists_lmmd else 0:.1f}±{avg_lmmd:.1f})')
fig.suptitle('融合特征 t-SNE 可视化 — 跨工况对齐效果对比 (FD001→FD002)\n'
             '目标域按 RUL 着色, 源域为深灰参考, 彩色椭圆=5个RUL子域分布, 虚线=子域质心距均值',
             fontsize=14,fontweight='bold',y=1.04)
fig.text(0.5,-0.01,summary,ha='center',fontsize=9,fontstyle='italic',color=C['dark'])

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR,'ch5_tsne_transfer.png'),dpi=300,bbox_inches='tight',facecolor='white')
plt.close(fig)
print(f"图7 已保存: ch5_tsne_transfer.png")
print(f"  子域质心距均值: 无迁移={avg_no:.1f}, 全局MMD={avg_mmd:.1f}, LMMD={avg_lmmd:.1f}")
print(f"  各子域MMD: {[f'{d:.1f}' for d in dists_mmd]}")
print(f"  各子域LMMD: {[f'{d:.1f}' for d in dists_lmmd]}")

# ================================================================
# 图8: 预测 vs 真实 RUL
# ================================================================
print("\n"+"="*60)
print("  图8: 预测 vs 真实 RUL 散点图")
print("="*60)

# ================================================================
# 加载全部 5 个模型并推理
# ================================================================
models_to_eval = {}  # {display_name: (preds, rmse, score, params)}

# --- LSTM ---
print("\n加载 LSTM...")
m=BasicLSTM(input_dim=NUM_FEATURES,hidden_dim=128,num_layers=3,dropout=0.3).to(DEVICE)
m.load_state_dict(torch.load(os.path.join(MODEL_DIR,'lstm_best_FD001.pt'),map_location=DEVICE,weights_only=False)['model_state_dict'])
p=predict_generic(m,X_fd001)
models_to_eval['BasicLSTM']=(p,compute_rmse(p,y_fd001),compute_nasa_score(p,y_fd001),sum(pp.numel() for pp in m.parameters()))

# --- GRU ---
print("加载 GRU...")
m=GRUModel(input_dim=NUM_FEATURES,hidden_dim=128,num_layers=3,dropout=0.3).to(DEVICE)
m.load_state_dict(torch.load(os.path.join(MODEL_DIR,'gru_best_FD001.pt'),map_location=DEVICE,weights_only=False)['model_state_dict'])
p=predict_generic(m,X_fd001)
models_to_eval['GRU']=(p,compute_rmse(p,y_fd001),compute_nasa_score(p,y_fd001),sum(pp.numel() for pp in m.parameters()))

# --- TCN ---
print("加载 TCN...")
m=TCNModel(input_dim=NUM_FEATURES,num_channels=64,kernel_size=3,num_layers=4,dropout=0.3).to(DEVICE)
m.load_state_dict(torch.load(os.path.join(MODEL_DIR,'tcn_best_FD001.pt'),map_location=DEVICE,weights_only=False)['model_state_dict'])
p=predict_generic(m,X_fd001)
models_to_eval['TCN']=(p,compute_rmse(p,y_fd001),compute_nasa_score(p,y_fd001),sum(pp.numel() for pp in m.parameters()))

# --- CNN+LSTM ---
print("加载 CNN+LSTM...")
m=CNN_LSTM_Model(input_dim=NUM_FEATURES,cnn_channels=64,lstm_hidden=64,lstm_layers=2,dropout=0.3).to(DEVICE)
m.load_state_dict(torch.load(os.path.join(MODEL_DIR,'cnn_lstm_best_FD001.pt'),map_location=DEVICE,weights_only=False)['model_state_dict'])
p=predict_generic(m,X_fd001)
models_to_eval['CNN+LSTM']=(p,compute_rmse(p,y_fd001),compute_nasa_score(p,y_fd001),sum(pp.numel() for pp in m.parameters()))

# --- STGNN (v2) ---
print("加载 STGNN (MSTCN+GAT)...")
m=build_stgnn_v2()
m.load_state_dict(torch.load(os.path.join(MODEL_DIR,'stgnn_v2_best_FD001.pt'),map_location=DEVICE,weights_only=False)['model_state_dict'])
p=predict_stgnn(m,X_fd001,edge_fd001)
models_to_eval['STGNN']=(p,compute_rmse(p,y_fd001),compute_nasa_score(p,y_fd001),sum(pp.numel() for pp in m.parameters()))

# 打印汇总
print(f"\n{'─'*70}")
print(f"{'模型':<16} {'RMSE':>8} {'Score':>12} {'Params':>10}")
print(f"{'─'*70}")
max_pred=0
for name,(preds,rmse,score,params) in models_to_eval.items():
    print(f"{name:<16} {rmse:>8.2f} {score:>12.1f} {params:>10,}")
    max_pred=max(max_pred,preds.max())
print(f"{'─'*70}")

# ================================================================
# 图8: 五模型融合散点图 — 单张 Axes
# ================================================================
# 配色 + 标记方案（5种颜色×5种标记，确保黑白打印也可区分）
model_styles=[
    ('BasicLSTM',     '#D62728', 'o',  'LSTM'),          # 红色圆点
    ('GRU',           '#FF7F0E', 's',  'GRU'),            # 橙色方块
    ('TCN',           '#2CA02C', 'D',  'TCN'),            # 绿色菱形
    ('CNN+LSTM',      '#9467BD', '^',  'CNN+LSTM'),       # 紫色三角
    ('STGNN',         '#1F77B4', 'v',  'STGNN (MSTCN+GAT)'), # 蓝色倒三角
]

fig,ax=plt.subplots(figsize=(10,8.5))

# 先画对角线（一次）
max_rul=max(y_fd001.max(),max_pred)*1.05
ax.plot([0,max_rul],[0,max_rul],'k--',linewidth=2.2,alpha=0.6,zorder=0,label='Ideal y=x (理想预测)')

# 逐模型画散点
for model_key,color,marker,legend_name in model_styles:
    preds,rmse,score,params=models_to_eval[model_key]
    ax.scatter(y_fd001.flatten(),preds.flatten(),
               c=color,marker=marker,alpha=0.35,s=28,
               edgecolors='white',linewidth=0.3,
               label=f'{legend_name}  (RMSE={rmse:.1f}, Score={score:.0f})',
               zorder=2)

ax.set_xlim(-2,max_rul);ax.set_ylim(-2,max_rul);ax.set_aspect('equal')
ax.set_xlabel('True RUL / 真实剩余寿命 (cycles)',fontsize=12,fontweight='bold')
ax.set_ylabel('Predicted RUL / 预测剩余寿命 (cycles)',fontsize=12,fontweight='bold')
ax.set_title('Predicted vs True RUL on FD001 Test Set\nFD001 测试集五模型预测对比',
             fontsize=14,fontweight='bold',pad=12)

# 图例外置（右侧）
legend=ax.legend(loc='upper left',fontsize=8.5,framealpha=0.9,
                 title='Model (RMSE / Score)',title_fontsize=9,
                 markerscale=1.3,handletextpad=0.6)
legend.get_frame().set_linewidth(1.2)

# 加 ±20 误差带半透明背景
ax.fill_between([0,max_rul],[-20,max_rul-20],[20,max_rul+20],
                alpha=0.04,color='gray',zorder=0)
ax.text(max_rul*0.98,max_rul*0.06,'Shaded: ±20 error band',
        ha='right',fontsize=8,fontstyle='italic',color=C['gray'])

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR,'ch5_pred_vs_true_scatter.png'),dpi=300,bbox_inches='tight',facecolor='white')
plt.close(fig)
print("\n图8 已保存: ch5_pred_vs_true_scatter.png (五模型合并)")

# ================================================================
# 图8附加: 五张独立子图 — 每模型单独一张
# ================================================================
SCATTER_DIR=os.path.join(FIG_DIR,'ch5_pred_vs_true_scatter')
os.makedirs(SCATTER_DIR,exist_ok=True)

for model_key,color,marker,legend_name in model_styles:
    preds,rmse,score,params=models_to_eval[model_key]

    fig,ax=plt.subplots(figsize=(7.5,7))

    # 散点
    ax.scatter(y_fd001.flatten(),preds.flatten(),
               c=color,alpha=0.45,s=35,
               edgecolors='white',linewidth=0.4,zorder=3)

    # 对角线
    ax.plot([0,max_rul],[0,max_rul],'k--',linewidth=2.2,alpha=0.6,
            zorder=0,label='Ideal y=x')

    ax.set_xlim(-2,max_rul);ax.set_ylim(-2,max_rul);ax.set_aspect('equal')
    ax.set_xlabel('True RUL (cycles)',fontsize=11,fontweight='bold')
    ax.set_ylabel('Predicted RUL (cycles)',fontsize=11,fontweight='bold')
    ax.set_title(f'{legend_name}\nRMSE={rmse:.2f}  |  Score={score:.1f}  |  Params={params:,}',
                 fontsize=13,fontweight='bold',color=color,pad=12)

    # 统计框
    ax.text(0.97,0.06,
            f'RMSE = {rmse:.2f}\nNASA Score = {score:.1f}\nParameters = {params:,}',
            transform=ax.transAxes,fontsize=10,
            verticalalignment='bottom',horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5',facecolor='white',
                     edgecolor=color,alpha=0.9,linewidth=1.8))

    ax.grid(True,alpha=0.25,linestyle='--')
    ax.legend(loc='lower right',fontsize=9.5,framealpha=0.85)

    plt.tight_layout()
    safe_name=model_key.replace(' ','_').replace('+','_')
    fig.savefig(os.path.join(SCATTER_DIR,f'{safe_name}.png'),
                dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(f"  独立子图已保存: ch5_pred_vs_true_scatter/{safe_name}.png")

print("\n[完成] 全部图表完成!")
