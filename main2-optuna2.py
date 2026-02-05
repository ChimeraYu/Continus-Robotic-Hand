import pandas as pd
import numpy as np
import os
import sys
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# --- Optuna 相关 ---
import optuna

# ======================
# 用户参数配置（优先修改这里）
# ======================
# 1) 每一节近似柔性关节长度
L = 200

# 2) 引导线数量与 Optuna 超参数搜索空间
NUM_WIRES = 2
ALPHA_RANGE = (2.0, 100.0)
W0_RANGE = (-200.0, 200.0)

# 3) 数据与模型配置（按需修改）
DATA_FILE_PATH = r'E:\桌面\code\1.xlsx'
E_FILE_PATH = r'E:\桌面\code\E.xlsx'
OUTPUT_PATH = r'E:\桌面\code\results_main3.xlsx'
PCA_COMPONENTS = NUM_WIRES
CUTOFF_FREQUENCY = 0.1
SAMPLE_RATE = 1.0


# ------------------------
# 1. 常用功能函数
# ------------------------
def rotate_vector(vector, angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    return rotation_matrix @ vector


def generate_path(segment_length, angle_list):
    path = [(0, 0)]
    current_point = np.array([0, 0])
    direction_vector = np.array([0, 1])  # Initial direction along positive y-axis

    for angle in angle_list:
        direction_vector = rotate_vector(direction_vector, angle)
        new_point = current_point + segment_length * direction_vector
        path.append(tuple(new_point))
        current_point = new_point
    return path


def plot_paths_side_by_side(paths_1, labels_1, title_1,
                            paths_2, labels_2, title_2):
    fig, axs = plt.subplots(1, 2, figsize=(24, 12))

    colors_1 = plt.cm.get_cmap('tab20', len(paths_1))
    colors_2 = plt.cm.get_cmap('tab20', len(paths_2))

    # Left plot
    for i, (path, label) in enumerate(zip(paths_1, labels_1)):
        x_coords, y_coords = zip(*path)
        axs[0].plot(x_coords, y_coords, marker='o', color=colors_1(i), label=label)
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title(title_1)
    axs[0].grid(True)
    axs[0].axis('equal')
    axs[0].legend()

    # Right plot
    for i, (path, label) in enumerate(zip(paths_2, labels_2)):
        x_coords, y_coords = zip(*path)
        axs[1].plot(x_coords, y_coords, marker='o', color=colors_2(i), label=label)
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].set_title(title_2)
    axs[1].grid(True)
    axs[1].axis('equal')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def contains_nan(data):
    return any(np.isnan(val) for val in data)


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return [x / norm for x in v]


def compute_e_n1(l, w_1, w_2):
    x = -l
    y = w_1 - w_2
    e_n1 = [x, y]
    return normalize_vector(e_n1)


def compute_W_element(l, w_2, e_n1):
    v1 = [l / 2, w_2]
    return v1[0] * e_n1[1] - v1[1] * e_n1[0]


def construct_matrix_W(l, w_lists):
    if len(w_lists) == 0:
        raise ValueError("w_lists should contain at least one wire trajectory.")

    all_W_matrices = []
    for w in w_lists:
        W_column = []
        for i in range(len(w) - 1):
            w_1 = w[i]
            w_2 = w[i + 1]
            e_n1 = compute_e_n1(l, w_1, w_2)
            W_element = compute_W_element(l, w_2, e_n1)
            W_column.append(W_element)

        if len(W_column) > 0:
            all_W_matrices.append(W_column)

    if len(all_W_matrices) == 0:
        raise ValueError("W matrix construction failed. No valid columns found.")

    if len(all_W_matrices) == 1:
        W_matrix = np.expand_dims(all_W_matrices[0], axis=1)
    else:
        W_matrix = np.column_stack(all_W_matrices)
    return W_matrix


def calculate_discriminant(B1, B2, B3):
    return B2 ** 2 - 4 * B1 * B3


def solve_quadratic(B1, B2, B3, discriminant, eps=1e-10):
    """
    通用求根：优先按二次方程求解；当 B1≈0 时退化为一次方程。
    """
    if abs(B1) <= eps:
        if abs(B2) <= eps:
            return None, None
        linear_root = -B3 / B2
        return linear_root, linear_root

    if discriminant < -eps:
        return None, None

    discriminant = max(discriminant, 0.0)
    sqrt_d = np.sqrt(discriminant)
    root1 = (-B2 + sqrt_d) / (2 * B1)
    root2 = (-B2 - sqrt_d) / (2 * B1)
    return root1, root2


def calculate_w(V_col, L, w_0, max_abs_w=200):
    """
    由列向量 V_col 依次计算 w(i)，结果返回一个列表 [w_0, w_1, ..., w_n]
    """
    n = V_col.shape[0]
    w_list = [w_0]
    for i in range(1, n + 1):
        v = V_col[i - 1, 0]
        w_prev = w_list[-1]
        B1 = v ** 2 - (L ** 2 / 4)
        B2 = -((2 * v ** 2 + (L ** 2 / 2)) * w_prev)
        B3 = (v ** 2 * L ** 2) + (v ** 2 - (L ** 2 / 4)) * w_prev ** 2
        discriminant = calculate_discriminant(B1, B2, B3)
        root1, root2 = solve_quadratic(B1, B2, B3, discriminant)
        if root1 is not None and root2 is not None:
            # 比较哪个 root 距离 w_prev 更近
            if abs(root1 - w_prev) < abs(root2 - w_prev):
                w_new = root1
            else:
                w_new = root2
            # 简单限定 |w_new| <= max_abs_w
            if abs(w_new) <= max_abs_w:
                w_list.append(w_new)
            else:
                w_list.append(np.nan)
        else:
            w_list.append(np.nan)
    return w_list




def build_param_names(num_wires):
    alpha_names = [f"alpha_{i + 1}" for i in range(num_wires)]
    w0_names = [f"w_0_{i + 1}" for i in range(num_wires)]
    return alpha_names, w0_names


def suggest_wire_parameters(trial, alpha_names, w0_names):
    alpha_values = [trial.suggest_float(name, *ALPHA_RANGE, log=True) for name in alpha_names]
    w0_values = [trial.suggest_float(name, *W0_RANGE) for name in w0_names]
    return alpha_values, w0_values


def calculate_w_lists(V, l, w0_values):
    w_lists = []
    for wire_idx, w0 in enumerate(w0_values):
        w_list = calculate_w(V[:, wire_idx:wire_idx + 1], l, w0)
        if contains_nan(w_list):
            return None
        w_lists.append(w_list)
    return w_lists


def butterworth_filter(data, cutoff, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def sample_keypoints_from_path(path, num_keypoints=8):
    """
    从完整path中等间隔抽取 num_keypoints 个关键点
    """
    if len(path) <= 1:
        return np.array(path)
    indices = np.linspace(0, len(path) - 1, num=num_keypoints, dtype=int)
    keypoints = [path[i] for i in indices]
    return np.array(keypoints)


def calculate_angle_mse(theta_actual, theta_desired):
    """
    对两个 N x M 的角度矩阵做均方误差
    """
    return mean_squared_error(theta_desired, theta_actual)


def calculate_keypoints_distance_mse(theta_actual, theta_desired,
                                     segment_length=8, num_keypoints=5):
    """
    对每一列的角度序列生成路径，抽取关键点后，比较坐标差异
    """
    n_cols = theta_actual.shape[1]
    total_distance = 0.0
    count = 0

    for col_idx in range(n_cols):
        angle_list_actual = theta_actual[:, col_idx]
        angle_list_desired = theta_desired[:, col_idx]

        path_actual = generate_path(segment_length, angle_list_actual)
        path_desired = generate_path(segment_length, angle_list_desired)

        keypoints_actual = sample_keypoints_from_path(path_actual, num_keypoints)
        keypoints_desired = sample_keypoints_from_path(path_desired, num_keypoints)

        if len(keypoints_actual) != len(keypoints_desired):
            continue
        dist = np.mean(np.sum((keypoints_actual - keypoints_desired) ** 2, axis=1))
        total_distance += dist
        count += 1

    if count == 0:
        return 0.0
    return total_distance / count


def calculate_comprehensive_loss(theta_actual, theta_desired,
                                 w_theta=0.3, w_pos=0.7,
                                 segment_length=8, num_keypoints=5):
    """
    组合角度MSE和关键点坐标MSE
    """
    angle_mse = calculate_angle_mse(theta_actual, theta_desired)
    pos_mse = calculate_keypoints_distance_mse(
        theta_actual, theta_desired,
        segment_length=segment_length,
        num_keypoints=num_keypoints
    )
    loss = w_theta * angle_mse + w_pos * pos_mse
    return loss


# ---------------------
# 全局数据加载和预处理
# ---------------------
data = pd.read_excel(DATA_FILE_PATH, header=None).values  # (N, M)

# Step 1: 计算每行均值 (theta_r_bar)
theta_r_bar = np.mean(data, axis=1, keepdims=True)  # (N, 1)

# Step 2: 数据中心化
centered_data = data - theta_r_bar

# Step 3: PCA 降维 (60 -> 2)
pca = PCA(n_components=PCA_COMPONENTS)
sigma_M = pca.fit_transform(centered_data.T)  # (M, NUM_WIRES)
S_M = pca.components_.T  # (N, NUM_WIRES)

# Step 4: Butterworth 对 S_M 的每一列平滑滤波
S_M_smoothed = np.zeros_like(S_M)
for i in range(S_M.shape[1]):
    S_M_smoothed[:, i] = butterworth_filter(S_M[:, i], CUTOFF_FREQUENCY, SAMPLE_RATE)

# Step 5: 将 sigma_M 坐标平移至非负
min_values = np.min(sigma_M, axis=0)  # (NUM_WIRES,)
sigma_M_shifted = sigma_M - min_values

# ==== 关键补充：更新 theta_r_bar (黄色公式) ====
theta_r_bar_updated = theta_r_bar + S_M @ min_values.reshape(-1, 1)
theta_r_bar = theta_r_bar_updated

# Step 6: 加载对角矩阵 E
E_data = pd.read_excel(E_FILE_PATH, header=None).values
E = np.diag(E_data.flatten())
E_inv = np.linalg.inv(E)

if PCA_COMPONENTS != NUM_WIRES:
    raise ValueError("PCA_COMPONENTS must equal NUM_WIRES for alpha/W dimensions to match.")

ALPHA_NAMES, W0_NAMES = build_param_names(NUM_WIRES)


# ---------------------
# 训练(优化)部分 - 使用 Optuna
# ---------------------
def objective(trial):
    # 1) 超参数搜索空间（按引导线数量自动生成）
    alpha_values, w0_values = suggest_wire_parameters(trial, ALPHA_NAMES, W0_NAMES)

    try:
        # 2) 组装 alpha 矩阵
        alpha_local = np.diag(alpha_values)
        alpha_inv_local = np.linalg.inv(alpha_local)

        # 3) 根据公式计算 f_local & V_local
        #    f_local = alpha * sigma_M_shifted 的转置
        f_local = alpha_local @ sigma_M_shifted.T  # (NUM_WIRES, M)
        V_local = E @ S_M_smoothed @ alpha_inv_local  # (N, NUM_WIRES)

        # 4) 分别计算每根引导线对应的 SoP
        w_lists_local = calculate_w_lists(V_local, L, w0_values)
        if w_lists_local is None:
            return 1e9  # 若出现 NaN，视为无效解

        # 5) 构建 W 矩阵
        W_matrix_local = construct_matrix_W(L, w_lists_local)

        # 6) 计算最终的 theta_results
        #    theta_results = E_inv * W * f_local + theta_r_bar
        theta_results_local = E_inv @ W_matrix_local @ f_local + theta_r_bar

        # 7) 计算综合损失(与原始 data 的差距)
        w_theta = 0.2
        w_pos = 0.8
        loss_local = calculate_comprehensive_loss(
            theta_results_local,
            data,
            w_theta=w_theta,
            w_pos=w_pos,
            segment_length=8,
            num_keypoints=5
        )
        return loss_local

    except Exception as e:
        print(f"[Warning] Bad parameters => {e}")
        return 1e9


if __name__ == "__main__":

    # 1. 创建一个研究(Study)
    study = optuna.create_study(direction="minimize")
    # 2. 在此研究中进行采样搜索
    study.optimize(objective, n_trials=200, show_progress_bar=True)

    best_params = study.best_params
    print("\n===== Optuna Optimization Finished =====")
    print(f"Best Params: {best_params}")
    print(f"Best Loss: {study.best_value}")

    # ================================
    # 用最优参数计算并可视化
    # ================================
    alpha_values_best = [best_params[name] for name in ALPHA_NAMES]
    w0_values_best = [best_params[name] for name in W0_NAMES]

    alpha_best = np.diag(alpha_values_best)
    alpha_inv_best = np.linalg.inv(alpha_best)

    # f_best = alpha_best @ sigma_M_shifted.T
    f_best = alpha_best @ sigma_M_shifted.T  # (NUM_WIRES, M)
    V_best = E @ S_M_smoothed @ alpha_inv_best  # (N, NUM_WIRES)

    w_lists_best = calculate_w_lists(V_best, L, w0_values_best)
    if w_lists_best is None:
        raise RuntimeError("Best parameters generated invalid SoP trajectories.")
    W_matrix_best = construct_matrix_W(L, w_lists_best)

    theta_results_best = E_inv @ W_matrix_best @ f_best + theta_r_bar

    # 可视化对比：1)原始 data 与 2)优化后的结果
    paths_theta_results = []
    labels_theta_results = []
    segment_length = L

    for col_idx in range(theta_results_best.shape[1]):
        angle_list = theta_results_best[:, col_idx]
        path = generate_path(segment_length, angle_list)
        paths_theta_results.append(path)
        labels_theta_results.append(f"Pose {col_idx + 1}")

    paths_data_1 = []
    labels_data_1 = []
    for col_idx in range(data.shape[1]):
        angle_list = data[:, col_idx]
        path = generate_path(segment_length, angle_list)
        paths_data_1.append(path)
        labels_data_1.append(f"Column {col_idx + 1}")

    plot_paths_side_by_side(
        paths_1=paths_data_1,
        labels_1=labels_data_1,
        title_1="Paths from Original Data (1.xlsx)",
        paths_2=paths_theta_results,
        labels_2=labels_theta_results,
        title_2="Optimized Paths (Optuna)"
    )

    # ================
    # Step 7: Save results
    # ================
    output_path = OUTPUT_PATH
    with pd.ExcelWriter(output_path) as writer:
        # 如果希望与“基准文件”相同 Sheet 名，可以按下述方式写入：
        pd.DataFrame(S_M).to_excel(writer, sheet_name='S(M)', index=False, header=False)
        pd.DataFrame(sigma_M_shifted).to_excel(writer, sheet_name='Sigma(M)', index=False, header=False)

        # 这里将 theta_results_best 视为“重构后”的 theta
        pd.DataFrame(theta_results_best).to_excel(writer, sheet_name='Theta_Reconstructed', index=False, header=False)

        # 同样将更新后的 theta_r_bar 保存
        pd.DataFrame(theta_r_bar).to_excel(writer, sheet_name='theta_r_bar', index=False, header=False)

        # **新增**：保存 W_matrix_best (SoP) 到名为 "SoP" 的 Sheet
        pd.DataFrame(w_lists_best).to_excel(writer, sheet_name='SoP', index=False, header=False)

        # 6) **新增** 保存 f_best
        #    如果想和 Excel 表示更直观，可以转置 -> (M, 2)
        #    如果您更习惯原状 (2, M)，则直接写 f_best 即可。
        pd.DataFrame(f_best.T).to_excel(writer, sheet_name='f_local', index=False, header=False)

    print(f"Results saved to {output_path}")
    print("Done.")

