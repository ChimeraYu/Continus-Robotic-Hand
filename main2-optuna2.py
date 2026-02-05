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
    if len(w_lists) != 2:
        raise ValueError("w_lists should contain two arrays.")

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


def solve_quadratic(B1, B2, discriminant):
    if discriminant >= 0:
        root1 = (-B2 + np.sqrt(discriminant)) / (2 * B1)
        root2 = (-B2 - np.sqrt(discriminant)) / (2 * B1)
        return root1, root2
    else:
        return None, None


def calculate_w(V_col, L, w_0):
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
        root1, root2 = solve_quadratic(B1, B2, discriminant)
        if root1 is not None and root2 is not None:
            # 比较哪个 root 距离 w_prev 更近
            if abs(root1 - w_prev) < abs(root2 - w_prev):
                w_new = root1
            else:
                w_new = root2
            # 简单限定 |w_new| <= 200
            if abs(w_new) <= 200:
                w_list.append(w_new)
            else:
                w_list.append(np.nan)
        else:
            w_list.append(np.nan)
    return w_list


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
file_path = r'E:\桌面\code\1.xlsx'
data = pd.read_excel(file_path, header=None).values  # (N, M)

# Step 1: 计算每行均值 (theta_r_bar)
theta_r_bar = np.mean(data, axis=1, keepdims=True)  # (N, 1)

# Step 2: 数据中心化
centered_data = data - theta_r_bar

# Step 3: PCA 降维 (60 -> 2)
pca = PCA(n_components=2)
sigma_M = pca.fit_transform(centered_data.T)  # (M, 2)
S_M = pca.components_.T  # (N, 2)

# Step 4: Butterworth 对 S_M 的每一列平滑滤波
cutoff_frequency = 0.1
sample_rate = 1.0
S_M_smoothed = np.zeros_like(S_M)
for i in range(S_M.shape[1]):
    S_M_smoothed[:, i] = butterworth_filter(S_M[:, i], cutoff_frequency, sample_rate)

# Step 5: 将 sigma_M 坐标平移至非负
min_values = np.min(sigma_M, axis=0)  # (2,)
sigma_M_shifted = sigma_M - min_values

# ==== 关键补充：更新 theta_r_bar (黄色公式) ====
theta_r_bar_updated = theta_r_bar + S_M @ min_values.reshape(-1, 1)
theta_r_bar = theta_r_bar_updated

# Step 6: 加载对角矩阵 E
file_path_E = r'E:\桌面\code\E.xlsx'
E_data = pd.read_excel(file_path_E, header=None).values

# 转换单位：从 N·mm/deg 转换为 N·mm/rad
conversion_factor = np.pi / 180  # 从 deg 到 rad 的转换系数
E_data_rad = E_data * conversion_factor

# 构造对角矩阵 E
E = np.diag(E_data_rad.flatten())

# 计算矩阵的逆
E_inv = np.linalg.inv(E)

# 长度 L
L = 200


# ---------------------
# 训练(优化)部分 - 使用 Optuna
# ---------------------
def objective(trial):
    # 1) 超参数搜索空间
    alpha_11 = trial.suggest_float("alpha_11", 0.000001, 0.01, log=True)
    alpha_22 = trial.suggest_float("alpha_22", 0.000001, 0.01, log=True)
    w_0_1 = trial.suggest_float("w_0_1", -200.0, 200.0)
    w_0_2 = trial.suggest_float("w_0_2", -200.0, 200.0)

    try:
        # 2) 组装 alpha 矩阵
        alpha_local = np.diag([alpha_11, alpha_22])
        alpha_inv_local = np.linalg.inv(alpha_local)

        # 3) 根据公式计算 f_local & V_local
        #    f_local = alpha * sigma_M_shifted 的转置
        f_local = alpha_local @ sigma_M_shifted.T  # (2, M)
        V_local = E @ S_M_smoothed @ alpha_inv_local  # (N, 2)

        # 4) 分别计算 SoP_1, SoP_2
        sop_1_local = calculate_w(V_local[:, 0:1], L, w_0_1)
        if contains_nan(sop_1_local):
            return 1e9  # 若出现 NaN，视为无效解

        sop_2_local = calculate_w(V_local[:, 1:2], L, w_0_2)
        if contains_nan(sop_2_local):
            return 1e9

        # 5) 构建 W 矩阵
        w_lists_local = [sop_1_local, sop_2_local]
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
    alpha_11_best = best_params['alpha_11']
    alpha_22_best = best_params['alpha_22']
    w_0_1_best = best_params['w_0_1']
    w_0_2_best = best_params['w_0_2']

    alpha_best = np.diag([alpha_11_best, alpha_22_best])
    alpha_inv_best = np.linalg.inv(alpha_best)

    # f_best = alpha_best @ sigma_M_shifted.T
    f_best = alpha_best @ sigma_M_shifted.T  # (2, M)
    V_best = E @ S_M_smoothed @ alpha_inv_best  # (N, 2)

    sop_1_best = calculate_w(V_best[:, 0:1], L, w_0_1_best)
    sop_2_best = calculate_w(V_best[:, 1:2], L, w_0_2_best)
    w_lists_best = [sop_1_best, sop_2_best]
    W_matrix_best = construct_matrix_W(L, w_lists_best)

    theta_results_best = E_inv @ W_matrix_best @ f_best + theta_r_bar

    # 可视化对比：1)原始 data 与 2)优化后的结果
    paths_theta_results = []
    labels_theta_results = []
    segment_length = 200

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
    output_path = r'E:\桌面\code\results_main3.xlsx'
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

