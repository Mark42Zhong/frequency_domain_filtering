import cv2
import numpy as np
from img_read import *
import datetime
import time

def fill_zeros(img):
    M, N, channel = img.shape

    filled_img = np.zeros((2 * M, 2 * N, channel))
    filled_img[0:M, 0:N] = img

    return filled_img


# 傅里叶变换
def dft(img):
    M, N, channel = img.shape
    x, y = [], []
    for i in range(M):
        x.extend([i] * N)
    for i in range(M):
        y.extend([i for i in range(N)])

    x = np.reshape(np.array(x), (M, N))
    y = np.reshape(np.array(y), (M, N))

    F = np.zeros((M, N, channel), dtype=np.complex)

    for c in range(channel):
        print("第{}层".format(c))
        for u in range(M):
            start = time.perf_counter()
            print("第{}行".format(u))
            for v in range(N):
                F[u, v, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (u * x / M + v * y / N)))
            end = time.perf_counter()
            print('运行时间 : %s 秒' % (end - start))
    return F


def shift(F):
    M, N, channel = F.shape

    shifted_F = np.zeros((M, N, channel), dtype=np.complex)

    shifted_F[0:M // 2, 0:N // 2] = F[M // 2:M, N // 2:N]
    shifted_F[0:M // 2, N // 2:N] = F[M // 2:M, 0:N // 2]
    shifted_F[M // 2:M, 0:N // 2] = F[0:M // 2, N // 2:N]
    shifted_F[M // 2:M, N // 2:N] = F[0:M // 2, 0:N // 2]

    return shifted_F


def ishift(shifted_F):
    M, N, channel = shifted_F.shape

    F = np.zeros((M, N, channel), dtype=np.complex)

    F[0:M // 2, 0:N // 2] = shifted_F[M // 2:M, N // 2:N]
    F[0:M // 2, N // 2:N] = shifted_F[M // 2:M, 0:N // 2]
    F[M // 2:M, 0:N // 2] = shifted_F[0:M // 2, N // 2:N]
    F[M // 2:M, N // 2:N] = shifted_F[0:M // 2, 0:N // 2]

    return F


def idft(F):
    M, N, channel = F.shape
    x, y = [], []
    for i in range(M):
        x.extend([i] * N)
    for i in range(M):
        y.extend([i for i in range(N)])

    x = np.reshape(np.array(x), (M, N))
    y = np.reshape(np.array(y), (M, N))

    img = np.zeros((M, N, channel), dtype=np.complex)

    for c in range(channel):
        print("第{}层".format(c))
        for u in range(M):
            print("第{}行".format(u))
            for v in range(N):
                img[u, v, c] = np.sum(F[..., c] * np.exp(2j * np.pi * (u * x / M + v * y / N))) / (M * N)

    return img


# 低通滤波器
def low_pass_filter(img):
    r = 50
    rows, cols, channel = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, channel), np.uint8)
    mask[crow - r:crow + r, ccol - r:ccol + r] = 1
    img = img * mask
    return img


# 高通滤波器
def high_pass_filter(img):
    r = 10
    rows, cols, channel = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, channel), np.uint8)
    mask[crow - r:crow + r, ccol - r:ccol + r] = 0
    img = img * mask
    return img


def save_gray(img, path):
    # 将频域图转换成灰度图输出
    # gray = cv2.cvtColor(img.clip(0, 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    cv2.imwrite(path, img.clip(0, 255).astype('uint8'))


def filter_operation(img, filter):
    path = "./{}/".format(filter) + filter
    # 进行滤波操作
    filled_img = fill_zeros(img)
    cv2.imwrite(path + "_after_zero_filled.jpg", filled_img)

    F = dft(filled_img)
    save_gray(F, path + "_after_dft.jpg")
    # cv2.imwrite(filter + "傅里叶变换后得到的频域灰度图.jpg", F.clip(0, 255).astype('uint8'))

    shifted_F = shift(F)
    save_gray(shifted_F, path + "_shift.jpg")
    # cv2.imwrite(filter + "象限交换后得到的频域灰度图.jpg", shifted_F.clip(0, 255).astype('uint8'))

    if filter == "low_pass_filter":
        shifted_F = low_pass_filter(shifted_F)
    elif filter == "high_pass_filter":
        shifted_F = high_pass_filter(shifted_F)

    save_gray(shifted_F, path + "_after_filter.jpg")
    # cv2.imwrite(filter + "进行滤波操作后的频域灰度图.jpg", shifted_F.clip(0, 255).astype('uint8'))

    F = ishift(shifted_F)
    save_gray(F, path + "_ishift.jpg")
    # cv2.imwrite(filter + "将象限换回原来的样子.jpg", shifted_F.clip(0, 255).astype('uint8'))

    filled_img = idft(F)
    img = filled_img[0:img.shape[0], 0:img.shape[1]]
    # print("逆傅里叶变换得到的处理后的图像：")
    # cv2_imshow(img)

    cv2.imwrite(path + "_after_idft.jpg", img.clip(0, 255).astype('uint8'))


if __name__ == "__main__":
    start = time.perf_counter()

    img_read()
    img = cv2.imread(r'./test_img/src_img.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img[100:200, 80:180]
    # img = img[190:200, 170:180]
    cv2.imwrite("./test_img/small_img.jpg", img)

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    print(img.shape)

    # mkdir("low_pass_filter")
    # filter_operation(img, "low_pass_filter")
    mkdir("high_pass_filter")
    filter_operation(img, "high_pass_filter")

    end = time.perf_counter()
    print('运行时间 : %s 秒' % (end - start))

