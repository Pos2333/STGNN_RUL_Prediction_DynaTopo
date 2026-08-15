# ============================================================
# core_models/base_models.py —— 基线模型
# ============================================================
# TODO 2 阶段仅包含基础 LSTM 模型
# 后续 TODO 3 会新增 MSTCN / GAT / Transformer 等模块
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# 防死亡 ReLU 辅助函数
# ============================================================
def _init_fc_bias_positive(fc_module):
    """
    将全连接输出头最后一层 Linear 的 bias 初始化为正数。

    ⚠️ 防死亡 ReLU：若 bias 初始为负，最后一层 ReLU（保证 RUL >= 0）
    输入恒为负 → 输出恒为 0，梯度无法回传（死亡 ReLU），训练完全停滞。
    与 stgnn_static.py / stgnn_dynatopo.py 中的修复保持一致。
    """
    last_linear = None
    for layer in fc_module:
        if isinstance(layer, nn.Linear):
            last_linear = layer
    if last_linear is not None and last_linear.bias is not None:
        last_linear.bias.data.fill_(1.0)


# ============================================================
# 基础 LSTM 模型 —— 用于 RUL 预测的基线
# ============================================================
class BasicLSTM(nn.Module):
    """
    一个简单但完整的 LSTM 模型，用于 C-MAPSS RUL 预测基线

    结构:
        输入 [batch, window_size, num_features]
          ↓
        LSTM 层（可堆叠多层）
          ↓
        取最后一个时间步的隐状态
          ↓
        全连接层 → 输出 [batch, 1]（预测 RUL 值）
          ↓
        ReLU 激活（保证输出非负，因为 RUL >= 0）
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        """
        参数:
            input_dim:   输入特征维度（= NUM_FEATURES = 17）
            hidden_dim:  LSTM 隐藏层维度
            num_layers:  LSTM 堆叠层数
            dropout:     LSTM 层间 dropout（num_layers > 1 时生效）
        """
        super(BasicLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ---- LSTM 层 ----
        # batch_first=True: 输入输出格式为 [batch, seq_len, features]
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ---- 全连接输出头 ----
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),    # 输出单个 RUL 值
            nn.ReLU()                           # RUL 非负约束
        )
        _init_fc_bias_positive(self.fc)  # 防死亡 ReLU

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征 [batch_size, window_size, input_dim]
               例：[256, 30, 17]

        返回:
            output: 预测 RUL [batch_size, 1]
        """
        # LSTM 前向传播
        # lstm_out: [batch, window_size, hidden_dim]
        # (h_n, c_n): 最后一层最终隐状态
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的输出（包含整条序列的信息）
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim]

        # 通过全连接层得到 RUL 预测值
        output = self.fc(last_hidden)  # [batch, 1]

        return output


# ============================================================
# 基础 GRU 模型 —— 用于 RUL 预测的基线（TODO 2.1）
# ============================================================
class GRUModel(nn.Module):
    """
    一个简单但完整的 GRU（门控循环单元）模型，作为 LSTM 的对比基线

    结构:
        输入 [batch, window_size, num_features]
          ↓
        GRU 层（可堆叠多层）
          ↓
        取最后一个时间步的隐状态
          ↓
        全连接层 → 输出 [batch, 1]（预测 RUL 值）
          ↓
        ReLU 激活（保证输出非负，因为 RUL >= 0）

    注意：所有超参数与 BasicLSTM 完全一致，确保公平对比。
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        """
        参数:
            input_dim:   输入特征维度（= NUM_FEATURES = 17）
            hidden_dim:  GRU 隐藏层维度
            num_layers:  GRU 堆叠层数
            dropout:     GRU 层间 dropout（num_layers > 1 时生效）
        """
        super(GRUModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ---- GRU 层 ----
        # batch_first=True: 输入输出格式为 [batch, seq_len, features]
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ---- 全连接输出头（与 LSTM 完全相同的结构）----
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),    # 输出单个 RUL 值
            nn.ReLU()                           # RUL 非负约束
        )
        _init_fc_bias_positive(self.fc)  # 防死亡 ReLU

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征 [batch_size, window_size, input_dim]
               例：[256, 30, 17]

        返回:
            output: 预测 RUL [batch_size, 1]
        """
        # GRU 前向传播
        # gru_out: [batch, window_size, hidden_dim]
        # h_n: 最后一层最终隐状态（GRU 只有一个隐状态，不像 LSTM 有 h 和 c）
        gru_out, _ = self.gru(x)

        # 取最后一个时间步的输出（包含整条序列的信息）
        last_hidden = gru_out[:, -1, :]  # [batch, hidden_dim]

        # 通过全连接层得到 RUL 预测值
        output = self.fc(last_hidden)  # [batch, 1]

        return output


# ============================================================
# 标准 TCN 模型 —— 单尺度膨胀因果卷积（TODO 2.2）
# ============================================================
class TCNModel(nn.Module):
    """
    标准时间卷积网络（TCN），仅使用单一卷积核尺寸（kernel_size=3），
    通过膨胀因子递增来扩大感受野，用于与多尺度 MSTCN 做对比，
    证明多尺度分支的有效性。

    结构:
        输入 [batch, W, N]  （W=窗口长度, N=特征数=17）
          ↓ permute
        [batch, N, W]  →  4 层膨胀因果 Conv1d（d=1,2,4,8）
          ↓ 全局平均池化（压缩时间维）
        [batch, 64]  →  全连接层 → [batch, 1]

    感受野验证:
        kernel_size=3, dilations=[1,2,4,8]（4层）
        RF = 1 + (3-1) * (1+2+4+8) = 1 + 2*15 = 31 > 30（窗口长度） ✅

    注意：输入使用全部 17 个特征（3 操作参数 + 14 传感器），
         与 LSTM/GRU 保持一致，确保公平对比。
    """

    def __init__(self, input_dim=17, num_channels=64, kernel_size=3,
                 num_layers=4, dropout=0.3):
        """
        参数:
            input_dim:    输入特征维度（= NUM_FEATURES = 17）
            num_channels: 卷积输出通道数（= 64，与 MSTCN 时序部分可比）
            kernel_size:  卷积核大小（固定为 3，单尺度）
            num_layers:   膨胀卷积层数（4 层，dilation 按 2^i 递增）
            dropout:      Dropout 比率
        """
        super(TCNModel, self).__init__()

        self.num_channels = num_channels

        # ---- 构建膨胀卷积层堆栈 ----
        # dilation 按 2^i 递增：1, 2, 4, 8
        layers = []
        in_ch = input_dim  # 第一层输入 = 17 个特征

        for i in range(num_layers):
            dilation = 2 ** i
            # padding = dilation 使得 kernel=3 时输出长度与输入相同
            padding = dilation
            layers.append(nn.Conv1d(
                in_ch, num_channels, kernel_size,
                padding=padding, dilation=dilation
            ))
            layers.append(nn.BatchNorm1d(num_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = num_channels  # 后续层输入输出通道数相同

        self.conv_stack = nn.Sequential(*layers)

        # ---- 全连接输出头（与 LSTM/GRU 结构一致）----
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels // 2, 1),    # 输出单个 RUL 值
            nn.ReLU()                           # RUL 非负约束
        )
        _init_fc_bias_positive(self.fc)  # 防死亡 ReLU

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征 [batch_size, window_size, input_dim]
               例：[256, 30, 17]

        返回:
            output: 预测 RUL [batch_size, 1]
        """
        # ---- Step 1: 形状转换 [B, W, N] → [B, N, W] ----
        # Conv1d 期望输入为 [batch, channels, length]
        x = x.permute(0, 2, 1)  # [B, 17, 30]

        # ---- Step 2: 膨胀卷积堆栈 ----
        x = self.conv_stack(x)  # [B, 64, 30]

        # ---- Step 3: 全局平均池化，压缩时间维 ----
        x = torch.mean(x, dim=-1)  # [B, 64]

        # ---- Step 4: 全连接输出 ----
        output = self.fc(x)  # [B, 1]

        return output


# ============================================================
# CNN + LSTM 串行混合模型 —— 无图结构的时序基线（TODO 2.3）
# ============================================================
class CNN_LSTM_Model(nn.Module):
    """
    CNN + LSTM 串行混合模型：先由 1D-CNN 提取局部短时特征并压缩序列长度，
    再由 LSTM 捕捉长时依赖。该模型不含图结构，用于证明显式建模
    传感器关联的必要性。

    结构:
        输入 [batch, W, N]  （W=30, N=17）
          ↓ permute → [B, 17, W]
        Conv1d(17→64, k=3) → ReLU → MaxPool(2)  → [B, 64, 15]
        Conv1d(64→64, k=3) → ReLU → MaxPool(2)  → [B, 64, 7]
          ↓ permute → [B, 7, 64]
        LSTM(64, hidden=64, num_layers=2)
          ↓ 取最后时间步
        全连接层 → [batch, 1]

    注意：输入使用全部 17 个特征，与 LSTM/GRU/TCN 保持一致。
    """

    def __init__(self, input_dim=17, cnn_channels=64, lstm_hidden=64,
                 lstm_layers=2, dropout=0.3):
        """
        参数:
            input_dim:    输入特征维度（= NUM_FEATURES = 17）
            cnn_channels: CNN 输出通道数（= 64）
            lstm_hidden:  LSTM 隐藏层维度（= 64）
            lstm_layers:  LSTM 堆叠层数（= 2）
            dropout:      Dropout 比率
        """
        super(CNN_LSTM_Model, self).__init__()

        # ---- CNN 局部特征提取部分 ----
        # 第一层：17 → 64，保持长度 30，池化后 → 15
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        # 第二层：64 → 64，保持长度 15，池化后 → 7
        self.conv2 = nn.Sequential(
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )

        # ---- LSTM 长时依赖捕捉部分 ----
        self.lstm = nn.LSTM(
            input_size=cnn_channels,       # 64
            hidden_size=lstm_hidden,       # 64
            num_layers=lstm_layers,        # 2
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # ---- 全连接输出头（与 LSTM/GRU/TCN 结构一致）----
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1),    # 输出单个 RUL 值
            nn.ReLU()                           # RUL 非负约束
        )
        _init_fc_bias_positive(self.fc)  # 防死亡 ReLU

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征 [batch_size, window_size, input_dim]
               例：[256, 30, 17]

        返回:
            output: 预测 RUL [batch_size, 1]
        """
        # ---- Step 1: 形状转换 [B, W, N] → [B, N, W] ----
        # Conv1d 期望输入为 [batch, channels, length]
        x = x.permute(0, 2, 1)  # [B, 17, 30]

        # ---- Step 2: CNN 局部特征提取 ----
        x = self.conv1(x)  # [B, 64, 15]
        x = self.conv2(x)  # [B, 64, 7]

        # ---- Step 3: 形状转换 [B, C, L] → [B, L, C]，送入 LSTM ----
        x = x.permute(0, 2, 1)  # [B, 7, 64]

        # ---- Step 4: LSTM 长时依赖捕捉 ----
        lstm_out, _ = self.lstm(x)  # [B, 7, 64]

        # 取最后一个时间步的隐状态
        last_hidden = lstm_out[:, -1, :]  # [B, 64]

        # ---- Step 5: 全连接输出 ----
        output = self.fc(last_hidden)  # [B, 1]

        return output


# ============================================================
# 测试入口
# ============================================================
if __name__ == '__main__':
    print("🧪 基础 LSTM 模型自测")

    # 模拟输入: [batch=4, window=30, features=17]
    dummy_input = torch.randn(4, 30, 17)
    print(f"输入形状: {dummy_input.shape}")

    # 实例化模型
    model = BasicLSTM(input_dim=17, hidden_dim=128, num_layers=3, dropout=0.3)
    print(f"LSTM 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 前向传播测试
    output = model(dummy_input)
    print(f"LSTM 输出形状: {output.shape}")  # 期望: [4, 1]
    print(f"LSTM 输出值: {output.flatten().tolist()}")

    print("\n" + "=" * 60)
    print("🧪 基础 GRU 模型自测")

    model_gru = GRUModel(input_dim=17, hidden_dim=128, num_layers=3, dropout=0.3)
    print(f"GRU 模型参数量: {sum(p.numel() for p in model_gru.parameters()):,}")
    output_gru = model_gru(dummy_input)
    print(f"GRU 输出形状: {output_gru.shape}")
    print(f"GRU 输出值: {output_gru.flatten().tolist()}")

    print("\n" + "=" * 60)
    print("🧪 标准 TCN 模型自测")

    model_tcn = TCNModel(input_dim=17, num_channels=64, kernel_size=3,
                         num_layers=4, dropout=0.3)
    print(f"TCN 模型参数量: {sum(p.numel() for p in model_tcn.parameters()):,}")
    output_tcn = model_tcn(dummy_input)
    print(f"TCN 输出形状: {output_tcn.shape}")
    print(f"TCN 输出值: {output_tcn.flatten().tolist()}")

    print("\n" + "=" * 60)
    print("🧪 CNN+LSTM 混合模型自测")

    model_cnn_lstm = CNN_LSTM_Model(input_dim=17, cnn_channels=64,
                                     lstm_hidden=64, lstm_layers=2, dropout=0.3)
    print(f"CNN+LSTM 模型参数量: {sum(p.numel() for p in model_cnn_lstm.parameters()):,}")
    output_cnn_lstm = model_cnn_lstm(dummy_input)
    print(f"CNN+LSTM 输出形状: {output_cnn_lstm.shape}")
    print(f"CNN+LSTM 输出值: {output_cnn_lstm.flatten().tolist()}")
