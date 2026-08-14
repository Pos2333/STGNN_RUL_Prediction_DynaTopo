# ============================================================
# utils/data_processor.py —— C-MAPSS 数据预处理管线
# ============================================================
# 功能：
#   1. 加载原始 txt 数据
#   2. 剔除无用特征（根据 config 中指定的索引）
#   3. Min-Max 归一化（train 上 fit，test 上仅 transform，防泄漏）
#   4. 滑动窗口样本构造
#   5. 分段线性 RUL 标签（上限截断）
#   6. 图结构构建（Spearman 相关 → edge_index）
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import torch
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    KEPT_SENSOR_INDICES,
    OP_SETTING_INDICES,
    ALL_FEATURE_INDICES,
    NUM_FEATURES,
    NUM_SENSORS,
    WINDOW_SIZE,
    RUL_CLIP_MAX,
    GRAPH_THRESHOLD
)


class CMAPSSDataProcessor:
    """
    C-MAPSS 数据预处理器

    使用方式：
        processor = CMAPSSDataProcessor()
        train_data = processor.process_train('FD001')
        test_data = processor.process_test('FD001')
    """

    def __init__(self, data_dir='data/raw', processed_dir='data/processed'):
        """
        初始化预处理器

        参数：
            data_dir: 原始数据所在目录
            processed_dir: 预处理后数据保存目录
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # 归一化到 [-1, 1]
        self.is_fitted = False  # 标记 scaler 是否已 fit（防止 test 先于 train 调用）

        # 确保输出目录存在
        os.makedirs(processed_dir, exist_ok=True)

    # ============================================================
    # 第1步：加载原始数据
    # ============================================================
    def load_raw_data(self, subset, data_type='train'):
        """
        加载原始 C-MAPSS 数据文件

        参数：
            subset: 数据集编号，如 'FD001', 'FD002' 等
            data_type: 'train' 或 'test'

        返回：
            df: 原始 DataFrame（26 列）
        """
        filename = f"{data_type}_{subset}.txt"
        filepath = os.path.join(self.data_dir, filename)

        # 原始数据以空格分隔，无表头
        df = pd.read_csv(filepath, sep=r'\s+', header=None)

        print(f"📂 已加载 {filename}，形状: {df.shape}")
        return df

    # ============================================================
    # 第2步：剔除无用特征
    # ============================================================
    def drop_useless_features(self, df):
        """
        根据 config 中的 ALL_FEATURE_INDICES 保留有用列
        丢弃 unit, cycle 以及方差为 0 的无效传感器

        参数：
            df: 原始 DataFrame（26 列）

        返回：
            features: 保留的特征矩阵 numpy [n_samples, NUM_FEATURES]
            unit_cycle: unit 和 cycle 列 [n_samples, 2]
        """
        # 提取 unit 和 cycle（列 0, 1），后续构造标签和窗口需要用
        unit_cycle = df.iloc[:, [0, 1]].values.astype(np.float32)
        # 提取有用特征
        features = df.iloc[:, ALL_FEATURE_INDICES].values.astype(np.float32)

        print(f"  ✅ 特征筛选完成：{df.shape[1]} 列 → {features.shape[1]} 列")
        return features, unit_cycle

    # ============================================================
    # 第3步：Normalization（Min-Max 归一化）
    # ============================================================
    def normalize(self, features, fit=True):
        """
        Min-Max 归一化到 [-1, 1]

        ⚠️ 极其重要：
           - 训练集调用 fit=True（fit + transform）
           - 测试集调用 fit=False（只用训练集的 min/max 做 transform）
           - 绝对防止数据泄露！

        参数：
            features: 特征矩阵 [n_samples, NUM_FEATURES]
            fit: 是否在此数据上拟合 scaler

        返回：
            normalized: 归一化后的特征
        """
        if fit:
            normalized = self.scaler.fit_transform(features)
            self.is_fitted = True
            print(f"  ✅ 归一化完成（fit + transform），范围: [{normalized.min():.2f}, {normalized.max():.2f}]")
        else:
            if not self.is_fitted:
                raise RuntimeError("❌ Scaler 尚未在训练集上拟合！请先调用 process_train()。")
            normalized = self.scaler.transform(features)
            print(f"  ✅ 归一化完成（仅 transform，使用训练集参数）")

        return normalized

    # ============================================================
    # 第4步：滑动窗口样本构造
    # ============================================================
    def build_sliding_windows(self, features, unit_cycle):
        """
        使用滑动窗口将时序数据切分为多个 (窗口, 标签) 样本

        原理：对每个发动机，按 window_size 逐时间步滑动，
        取 window_size 个时间步的特征作为 X，最后一个时间步的 RUL 作为 y

        参数：
            features: 归一化后的特征 [n_total_samples, NUM_FEATURES]
            unit_cycle: [n_total_samples, 2] -> [unit_id, cycle]

        返回：
            X: 窗口特征 [n_samples, WINDOW_SIZE, NUM_FEATURES]
            y: RUL 标签 [n_samples]
            unit_ids: [n_samples] 每个窗口样本所属的发动机编号
                      （用于按发动机分组拆分训练/验证集，防止数据泄漏）
        """
        X_list = []
        y_list = []
        unit_ids_list = []

        # 获取所有发动机编号
        unique_units = np.unique(unit_cycle[:, 0])

        for unit_id in unique_units:
            # 定位该发动机的所有样本
            mask = unit_cycle[:, 0] == unit_id
            unit_features = features[mask]         # [life_cycles, NUM_FEATURES]
            unit_cycles = unit_cycle[mask, 1]       # [life_cycles]

            life_length = len(unit_features)

            # 如果发动机运行周期 < 窗口大小，跳过（数据不足）
            if life_length < WINDOW_SIZE:
                continue

            # 滑动窗口切分
            for i in range(life_length - WINDOW_SIZE + 1):
                # 取 [i, i+WINDOW_SIZE-1] 共 WINDOW_SIZE 个时间步的特征
                window_features = unit_features[i:i + WINDOW_SIZE]
                # 标签：窗口最后一个时间步的 RUL
                rul_label = life_length - (i + WINDOW_SIZE)

                X_list.append(window_features)
                y_list.append(rul_label)
                unit_ids_list.append(unit_id)

        X = np.stack(X_list, axis=0)  # [n_samples, WINDOW_SIZE, NUM_FEATURES]
        y = np.array(y_list)          # [n_samples]
        unit_ids = np.array(unit_ids_list, dtype=np.float32)  # [n_samples]

        print(f"  ✅ 滑动窗口构造完成：{len(X)} 个样本（来自 {len(unique_units)} 台发动机）")
        print(f"     窗口形状: {X.shape}, 标签形状: {y.shape}")
        return X, y, unit_ids

    # ============================================================
    # 第5步：分段线性 RUL 标签（上限截断）
    # ============================================================
    def clip_rul_labels(self, y):
        """
        将 RUL 标签截断到上限 RUL_CLIP_MAX（默认 125）

        原理：发动机早期退化不明显，RUL 变化不线性。
        将 > RUL_CLIP_MAX 的标签统一截断为 RUL_CLIP_MAX，
        让模型专注于预测最近 125 个周期的退化趋势。

        参数：
            y: RUL 标签数组

        返回：
            y_clipped: 截断后的标签
        """
        y_clipped = np.clip(y, a_min=0, a_max=RUL_CLIP_MAX)
        print(f"  ✅ RUL 标签截断完成（上限 = {RUL_CLIP_MAX}）")
        print(f"     截断前范围: [{y.min():.0f}, {y.max():.0f}]")
        print(f"     截断后范围: [{y_clipped.min():.0f}, {y_clipped.max():.0f}]")
        return y_clipped

    # ============================================================
    # 第6步：图结构构建
    # ============================================================
    def build_graph_structure(self, features):
        """
        基于传感器之间的 Spearman 秩相关系数构建图邻接矩阵

        步骤：
          1. 计算 NUM_SENSORS 个传感器（不含操作参数）的 Spearman 相关矩阵
          2. 取绝对值，按阈值 GRAPH_THRESHOLD 二值化（高于阈值则有边）
          3. 转为 PyG 的 edge_index 格式 [2, num_edges]

        参数：
            features: 归一化后的特征 [n_total_samples, NUM_FEATURES]
                      （前 3 列是 op1~op3，后 14 列是保留传感器）

        返回：
            edge_index: PyG 格式的边索引 [2, num_edges]
            edge_weight: 边的权重（Spearman 相关系数绝对值）[num_edges]
        """
        # 仅取传感器部分（跳过 op1, op2, op3）
        sensor_data = features[:, 3:]  # 后 14 列是传感器

        # 计算 Spearman 秩相关系数矩阵
        corr_matrix, _ = spearmanr(sensor_data)
        corr_abs = np.abs(corr_matrix)  # 取绝对值

        # 二值化：相关系数 > 阈值则有边
        adj_matrix = (corr_abs > GRAPH_THRESHOLD).astype(np.float32)

        # 去掉自环（对角线设为 0）
        np.fill_diagonal(adj_matrix, 0)

        # 转换为 PyG 的 edge_index 格式 [2, num_edges]
        edge_index = np.array(np.where(adj_matrix > 0))  # [2, num_edges]
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # 提取边权重
        edge_weight = corr_abs[adj_matrix > 0]
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

        num_edges = edge_index.shape[1]
        print(f"  ✅ 图结构构建完成：{num_edges} 条边（共 {NUM_SENSORS} 个节点）")
        print(f"     阈值: {GRAPH_THRESHOLD}, 平均度数: {num_edges / NUM_SENSORS:.1f}")

        return edge_index, edge_weight

    # ============================================================
    # 完整训练数据预处理流程
    # ============================================================
    def process_train(self, subset='FD001', save=True):
        """
        完整的训练数据预处理管线

        参数：
            subset: 数据集编号
            save: 是否保存预处理后的数据到 processed_dir

        返回：
            X: 窗口特征 [n_samples, WINDOW_SIZE, NUM_FEATURES]
            y: RUL 标签 [n_samples]
            unit_ids: 每个窗口样本所属的发动机编号 [n_samples]
            edge_index: 图边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges]
        """
        print(f"\n{'='*60}")
        print(f"  预处理训练集：{subset}")
        print(f"{'='*60}")

        # Step 1: 加载数据
        df = self.load_raw_data(subset, data_type='train')

        # Step 2: 剔除无用特征
        features, unit_cycle = self.drop_useless_features(df)

        # Step 3: 归一化（fit + transform）
        features = self.normalize(features, fit=True)

        # Step 4: 滑动窗口
        X, y, unit_ids = self.build_sliding_windows(features, unit_cycle)

        # Step 5: RUL 标签截断
        y = self.clip_rul_labels(y)

        # Step 6: 图结构构建
        edge_index, edge_weight = self.build_graph_structure(features)

        # 保存到 processed 目录
        if save:
            save_path = os.path.join(self.processed_dir, f'{subset}_train')
            # ⚠️ unit 字段必须保存：按发动机（unit）分组拆分训练/验证集时使用，
            #    防止同一台发动机的窗口样本被同时分进训练集和验证集（数据泄漏）
            np.savez(save_path, X=X, y=y, unit=unit_ids)
            torch.save({'edge_index': edge_index, 'edge_weight': edge_weight},
                       f'{save_path}_graph.pt')
            print(f"  💾 已保存到 {save_path}.npz 和 {save_path}_graph.pt")

        return X, y, unit_ids, edge_index, edge_weight

    # ============================================================
    # 完整测试数据预处理流程（⭐防泄漏关键）
    # ============================================================
    def process_test(self, subset='FD001', save=True):
        """
        完整的测试数据预处理管线
        ⚠️ 使用训练集的 scaler 参数，绝对不 fit 测试数据！

        参数：
            subset: 数据集编号
            save: 是否保存

        返回：
            X: 窗口特征 [n_samples, WINDOW_SIZE, NUM_FEATURES]
            y: RUL 标签 [n_samples]
            rul_true: 原始 RUL 真值 [n_samples]（用于评估）
        """
        print(f"\n{'='*60}")
        print(f"  预处理测试集：{subset}")
        print(f"{'='*60}")

        # Step 1: 加载数据
        df_test = self.load_raw_data(subset, data_type='test')

        # Step 2: 剔除无用特征
        features_test, unit_cycle_test = self.drop_useless_features(df_test)

        # Step 3: 归一化（仅 transform，使用 train 的参数！）
        if not self.is_fitted:
            raise RuntimeError("❌ Scaler 尚未在训练集上拟合！请先调用 process_train()。")
        features_test = self.normalize(features_test, fit=False)

        # Step 4: 加载真实 RUL 标签
        rul_df = pd.read_csv(os.path.join(self.data_dir, f'RUL_{subset}.txt'),
                             sep=r'\s+', header=None)
        true_rul = rul_df.values.flatten().astype(np.float32)

        # Step 5: 对每个测试发动机构造窗口样本
        X_list, y_list = [], []

        unique_units = np.unique(unit_cycle_test[:, 0])
        for idx, unit_id in enumerate(unique_units):
            mask = unit_cycle_test[:, 0] == unit_id
            unit_features = features_test[mask]    # [life_cycles, NUM_FEATURES]

            life_length = len(unit_features)

            if life_length < WINDOW_SIZE:
                continue

            # 只取最后一个窗口（测试时只需预测最后一个时间点的 RUL）
            window = unit_features[-WINDOW_SIZE:]  # [WINDOW_SIZE, NUM_FEATURES]
            X_list.append(window)
            y_list.append(true_rul[idx])

        X = np.stack(X_list, axis=0)  # [n_engines, WINDOW_SIZE, NUM_FEATURES]
        y = np.array(y_list)          # [n_engines]

        print(f"  ✅ 测试集窗口构造完成：{len(X)} 个发动机样本")
        print(f"     窗口形状: {X.shape}, 标签形状: {y.shape}")

        # Step 6: RUL 标签截断（与训练一致）
        y = self.clip_rul_labels(y)

        # 保存
        if save:
            save_path = os.path.join(self.processed_dir, f'{subset}_test')
            np.savez(save_path, X=X, y=y, true_rul=true_rul)
            print(f"  💾 已保存到 {save_path}.npz")

        return X, y, true_rul


# ============================================================
# 便捷函数：按发动机（unit）分组拆分训练/验证集
# ============================================================
def split_by_unit(X, y, unit_ids, val_ratio=0.2, random_state=42):
    """
    按发动机（unit）分组拆分训练集/验证集，防止数据泄漏。

    ⚠️ 铁律：同一台发动机的所有窗口样本必须进同一边（训练或验证），
      绝不能跨两边。样本级随机拆分（train_test_split）会把同一台
      发动机的样本同时分进训练集和验证集，导致验证集泄漏
      （val_rmse 虚假偏低、与 test_rmse 差 3 倍）。

    使用 sklearn 的 GroupShuffleSplit：
      - 以 unit id 为分组依据，按"发动机台数"比例拆分（如 0.2 → 20% 台发动机进验证）
      - random_state 固定，保证可复现

    参数:
        X:          窗口特征 [n_samples, WINDOW_SIZE, NUM_FEATURES]
        y:          RUL 标签 [n_samples]
        unit_ids:   每个窗口样本所属的发动机编号 [n_samples]
        val_ratio:  验证集比例（按发动机台数），默认 0.2
        random_state: 随机种子，默认 42（与 configs.config.RANDOM_SEED 一致）

    返回:
        X_train, X_val, y_train, y_val
    """
    from sklearn.model_selection import GroupShuffleSplit

    # unit_ids 可能为 float32（来自 unit_cycle），转成整数保证分组语义正确
    groups = np.asarray(unit_ids).astype(np.int64)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 打印分组统计，便于确认没有跨组泄漏
    train_units = np.unique(groups[train_idx])
    val_units = np.unique(groups[val_idx])
    overlap = np.intersect1d(train_units, val_units)
    print(f"  🧩 按发动机分组拆分: 训练 {len(X_train)} 样本/{len(train_units)} 台, "
          f"验证 {len(X_val)} 样本/{len(val_units)} 台")
    if len(overlap) > 0:
        raise RuntimeError(f"❌ 数据泄漏！{len(overlap)} 台发动机同时出现在训练和验证集: {overlap[:10]}")
    print(f"  ✅ 训练/验证发动机无重叠")

    return X_train, X_val, y_train, y_val


# ============================================================
# 便捷函数：一键预处理所有数据集
# ============================================================
def preprocess_all_datasets(data_dir='data/raw', processed_dir='data/processed'):
    """
    一键预处理 FD001~FD004 所有训练集和测试集

    返回：
        processed_data: 字典，包含所有预处理结果
    """
    processor = CMAPSSDataProcessor(data_dir=data_dir, processed_dir=processed_dir)
    results = {}

    for subset in ['FD001', 'FD002', 'FD003', 'FD004']:
        print(f"\n{'#'*60}")
        print(f"#  处理数据集：{subset}")
        print(f"{'#'*60}")

        # 处理训练集
        X_train, y_train, unit_ids, edge_index, edge_weight = processor.process_train(subset)
        results[f'{subset}_train'] = {
            'X': X_train, 'y': y_train, 'unit': unit_ids,
            'edge_index': edge_index, 'edge_weight': edge_weight
        }

        # 处理测试集
        X_test, y_test, true_rul = processor.process_test(subset)
        results[f'{subset}_test'] = {
            'X': X_test, 'y': y_test, 'true_rul': true_rul
        }

    print(f"\n{'='*60}")
    print(f"  🎉 全部数据集预处理完成！")
    print(f"{'='*60}")
    return results


# ============================================================
# 测试入口
# ============================================================
if __name__ == '__main__':
    # 快速测试：只处理 FD001
    print("🧪 数据预处理器自测模式")
    processor = CMAPSSDataProcessor()

    # 处理训练集
    X_train, y_train, unit_ids, edge_idx, edge_w = processor.process_train('FD001')

    # 处理测试集（自动使用训练集的 scaler）
    X_test, y_test, true_rul = processor.process_test('FD001')

    print(f"\n{'='*60}")
    print(f"  ✅ 自测通过！")
    print(f"  训练集: X={X_train.shape}, y={y_train.shape}, unit={unit_ids.shape}")
    print(f"  测试集: X={X_test.shape}, y={y_test.shape}")
    print(f"  图结构: {edge_idx.shape[1]} 条边")
    print(f"{'='*60}")
