# ============================================================
# configs/config.py —— 全局配置参数
# ============================================================
# 本文件中的传感器筛选结论来自 notebooks/data_exploration.ipynb
# 通过 RandomForest 在 FD001 上的特征重要性分析得出

# ============================================================
# 1. 数据预处理相关
# ============================================================

# 滑动窗口长度（时间步数）
WINDOW_SIZE = 30

# RUL 分段线性标签的上限（超过此值的发动机早期阶段，RUL 截断为该值）
RUL_CLIP_MAX = 125

# 保留的传感器在原始 26 列中的索引（0-based）
# 排除了方差为 0 的 7 个无效传感器:
#   sensor_1 (col5), sensor_5 (col9), sensor_6 (col10),
#   sensor_10 (col14), sensor_16 (col20), sensor_18 (col22), sensor_19 (col23)
KEPT_SENSOR_INDICES = [15, 13, 8, 16, 11, 18, 19, 25, 7, 6, 24, 17, 12, 21]

# 保留传感器名称（按重要性排序，由高到低）
KEPT_SENSOR_NAMES = [
    'sensor_11', 'sensor_9', 'sensor_4', 'sensor_12',
    'sensor_7', 'sensor_14', 'sensor_15', 'sensor_21',
    'sensor_3', 'sensor_2', 'sensor_20', 'sensor_13',
    'sensor_8', 'sensor_17'
]

# 是否保留操作参数（op1, op2, op3）
# FD001 中 op1~op3 方差接近 0，但在 FD002~FD004 多工况下会变化
# 设置在 data_processor 中明确处理
KEEP_OPERATIONAL_SETTINGS = True

# 操作参数列索引
OP_SETTING_INDICES = [2, 3, 4]  # op1, op2, op3（0-based）

# 最终特征列 = 操作参数 + 保留传感器
# 特征数量 = 3 + 14 = 17
ALL_FEATURE_INDICES = OP_SETTING_INDICES + KEPT_SENSOR_INDICES

# 最终特征矩阵列数
NUM_FEATURES = len(ALL_FEATURE_INDICES)  # 17

# 保留传感器数量（用于图结构大小）
NUM_SENSORS = len(KEPT_SENSOR_INDICES)  # 14

# ============================================================
# 2. 图结构构建相关
# ============================================================

# Spearman 相关矩阵转邻接矩阵的阈值（相关系数绝对值 > 此值则有边）
GRAPH_THRESHOLD = 0.6

# ============================================================
# 3. 训练相关
# ============================================================

# 批次大小
BATCH_SIZE = 256

# 初始学习率
LEARNING_RATE = 0.001

# 训练轮数
NUM_EPOCHS = 100

# 早停耐心值（验证 loss 不再下降的容忍轮数）
EARLY_STOP_PATIENCE = 20

# 随机种子（保证可复现）
RANDOM_SEED = 42

# ============================================================
# 4. 模型结构相关（STGNN）
# ============================================================

# MSTCN 相关
MSTCN_NUM_CHANNELS = [32, 64, 128]   # 各层通道数
MSTCN_KERNEL_SIZES = [3, 5, 7]       # 各层卷积核大小
MSTCN_DROPOUT = 0.2

# GAT 相关
GAT_HIDDEN_DIM = 64                   # GAT 隐藏层维度
GAT_HEADS = 4                         # 注意力头数
GAT_DROPOUT = 0.2

# Transformer 相关
TRANSFORMER_D_MODEL = 128             # Transformer 模型维度
TRANSFORMER_NHEAD = 4                 # 多头注意力头数
TRANSFORMER_NUM_LAYERS = 2            # Transformer Encoder 层数
TRANSFORMER_DROPOUT = 0.2

# 全连接层
FC_HIDDEN_DIM = 64                    # 最后全连接层维度

# ============================================================
# 5. 迁移学习相关
# ============================================================

# LMMD 损失权重（迁移学习时使用，TODO 5 才会用到）
LMMD_LAMBDA = 0.1

# 目标域任务损失权重（半监督迁移时，目标域 CombinedLoss 的系数）
# 设为 0 则退化为无监督 LMMD；设为 1.0 则源域和目标域等权
TGT_TASK_WEIGHT = 1.0

# 损失函数权重
MSE_WEIGHT = 0.5
NASA_SCORE_WEIGHT = 0.5
