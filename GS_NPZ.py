import os
import numpy.random as random
import numpy as np
from scipy.fft import fft2, ifft2


# 高斯滤波函数
def gs(amp_2, w):
    x, y = amp_2.shape
    x_center, y_center = x // 2, y // 2
    for i in range(x):
        for j in range(y):
            # 计算当前点到中心的距离的平方
            dist_sq = (i - x_center) ** 2 + (j - y_center) ** 2
            # 应用高斯权重
            amp_2[i, j] *= np.exp(-dist_sq / (2 * w ** 2))
    return amp_2


def gerchberg_saxton(object_amp, target_amp, num_iterations):
    object_phase = (2 * np.random.rand(*object_amp.shape) - 1) * np.pi
    for _ in range(num_iterations):
        object_field = object_amp * np.exp(1j * object_phase)
        target_field = fft2(object_field)
        temp_phase = np.angle(target_field)
        target_field = target_amp * np.exp(1j * temp_phase)
        object_field = ifft2(target_field)
        object_phase = np.angle(object_field)
    return object_phase


def generate_random(min_val, max_val, step=0.25):
    # 首先将范围除以步长得到调整后的最小值和最大值
    adjusted_min = int(min_val / step)
    adjusted_max = int(max_val / step)
    # 在调整后的范围内生成一个随机数
    random_adjusted = random.randint(adjusted_min, adjusted_max)
    # 将生成的整数乘以步长得到最终的随机数
    return random_adjusted * step


HG_path = os.path.join(f"T1.npz")
image = np.load(HG_path)['arr_0']
for i in range(0,5000):
    t = generate_random(20, 82, 0.25)
    amp_temp = np.ones_like(image)
    amp_2 = gs(amp_temp, t)
    phase_gs = gerchberg_saxton(amp_2, image, 100)
    image_recover = amp_2 * np.exp(1j * phase_gs)
    image_fft = np.abs(np.fft.fft2(image_recover))  # 取FFT的振幅并进行中心化
    image_fft = (image_fft - np.min(image_fft)) / (np.max(image_fft) - np.min(image_fft))
    Gs_dir = os.path.join("train_img/GS", f"{i}.npz")
    HG_dir = os.path.join("train_img/HG", f"{i}.npz")
    np.savez(Gs_dir, amp_2)
    np.savez(HG_dir, image_fft)
    print(f"{i}==>{t}==>{np.shape(amp_2)}")
