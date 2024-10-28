import os
import numpy.random as random
import numpy as np
from scipy.fft import fft2, ifft2


# ��˹�˲�����
def gs(amp_2, w):
    x, y = amp_2.shape
    x_center, y_center = x // 2, y // 2
    for i in range(x):
        for j in range(y):
            # ���㵱ǰ�㵽���ĵľ����ƽ��
            dist_sq = (i - x_center) ** 2 + (j - y_center) ** 2
            # Ӧ�ø�˹Ȩ��
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
    # ���Ƚ���Χ���Բ����õ����������Сֵ�����ֵ
    adjusted_min = int(min_val / step)
    adjusted_max = int(max_val / step)
    # �ڵ�����ķ�Χ������һ�������
    random_adjusted = random.randint(adjusted_min, adjusted_max)
    # �����ɵ��������Բ����õ����յ������
    return random_adjusted * step


HG_path = os.path.join(f"T1.npz")
image = np.load(HG_path)['arr_0']
for i in range(0,5000):
    t = generate_random(20, 82, 0.25)
    amp_temp = np.ones_like(image)
    amp_2 = gs(amp_temp, t)
    phase_gs = gerchberg_saxton(amp_2, image, 100)
    image_recover = amp_2 * np.exp(1j * phase_gs)
    image_fft = np.abs(np.fft.fft2(image_recover))  # ȡFFT��������������Ļ�
    image_fft = (image_fft - np.min(image_fft)) / (np.max(image_fft) - np.min(image_fft))
    Gs_dir = os.path.join("train_img/GS", f"{i}.npz")
    HG_dir = os.path.join("train_img/HG", f"{i}.npz")
    np.savez(Gs_dir, amp_2)
    np.savez(HG_dir, image_fft)
    print(f"{i}==>{t}==>{np.shape(amp_2)}")
