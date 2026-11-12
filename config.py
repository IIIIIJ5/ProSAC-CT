"""Experiment settings shared by train.py and test.py.
实验配置：train.py 与 test.py 共用。做消融实验时只需修改下面的模块开关。
"""

# ============================== 模块开关 / module switches ==============================
use_apgc = True    # APGC：中心 LDCT 切片的 MedDINOv3 解剖特征 × Sobel/Laplacian 边缘特征，注入共享编码器
use_rfdds = True   # RFDDS：在共享潜在表示上做低/中/高频解耦与通道级重加权（每个阶段一个）
use_td3 = True     # TD3：早/中/晚三个时间步阶段各用一个解码器；关闭后所有时间步共用一个解码器

# ============================== 数据 / data ==============================
data_dir = 'data/lists'   # 存放 {train,test}_{ndct,ldct,ldct_prev,ldct_next}.txt 的文件夹
dataroot = ''             # 拼接在列表中每条路径前面的公共前缀
image_size = 128          # 切片统一缩放到 image_size x image_size

# ============================== 训练 / training ==============================
results_dir = 'results/prosac_ct'
batch_size = 4
grad_accum = 1            # 梯度累积步数
lr = 2e-4
num_steps = 100000
ema_decay = 0.995
loss_weights = (1.0, 1.0, 0.1)  # (lambda_r, lambda_eps, lambda_x)：残差、噪声、重建图像三项 L2 损失的权重
eval_every = 5000         # 每隔多少步在测试集上评估一次
save_last_every = 5000    # 每隔多少步更新 last.pt（用于断点续训）
save_every = 100000       # 每隔多少步另存一个 model-<step>.pt
num_workers = 4
seed = 10

# ============================== 采样与测试 / sampling and testing ==============================
sampling_steps = 10       # 在 t = 999, 899, ..., 99 处评估网络，最后一步直接输出 x0 估计
test_batch_size = 16
