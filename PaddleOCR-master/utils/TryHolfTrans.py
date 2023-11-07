import cv2 as cv  # opencv-python==4.2.0.34
import numpy as np
import os
import os.path as osp
import time
import math
class HoughTransUtil:
  def HoughTrans(img, src, result, h, w):
    '''
    :param img: 二值化后的图像
    :param src: 原始图像
    :param result: 矫正图像保存路径
    :param h: 原始图像的高
    :param w: 原始图像的宽
    :return: 霍夫变换的阈值、旋转角度
    '''
    edges = cv.Canny(img, 50, 200, apertureSize=3)  # 边缘检测
    arg = 300
     # 霍夫变换检测直线，返回数组：（rho，theta）。rho以像素为单位测量，theta以弧度为单位测量。
    # while not isinstance(lines, np.ndarray):
    #     if arg <= 200:
    #         break
    #     arg -= 40
    #     lines = cv.HoughLines(edges, 1, np.pi / 180, arg)
    while arg>0:
        lines = cv.HoughLines(edges, 1, np.pi / 180, arg)
        if lines is None:
            arg = arg-20

    sum_theta = 0
    sum_0 = 0
    for line in lines:
        rho, theta = line[0]
        real_angel = -(theta * 180 / math.pi - 90)
        if -45 < real_angel < 45:
            sum_theta += theta
        else:
            sum_0 += 1
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(src, (x1, y1), (x2, y2), (0, 0, 255), 1)
    # 将所有直线画出
    cv.imshow('table/23_1.jpg', src)
    cv.waitKey()
    cv.destroyAllWindows()
    average_theta = sum_theta / len(lines)
    angle = average_theta * 180 / math.pi - 90
    if -45 < angle < 45:
        angle = -angle
    else:
        angle = 0
    center = (w // 2, h // 2)  # 矩形中心
    M = cv.getRotationMatrix2D(center, -angle, 1.0)  # 传入中心和角度，得到旋转矩形
    rotated = cv.warpAffine(src, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)  # 最后要换成原图
    cv.imwrite(result, rotated)
    return arg, angle

# if __name__ == '__main__':
#     filepath = 'image/'
#     result_path = 'result/'
#     start = time.time()
#     for file in os.listdir(filepath):
#         if is_img(osp.splitext(file)[1]):
#             print(file)
#             src = cv.imread(filepath + file)
#             src = cv.copyMakeBorder(src, 50, 50, 50, 50, cv.BORDER_CONSTANT, value=[255, 255, 255])
#             (h, w) = src.shape[:2]
#             binary_ = binary(src)  # 二值化、滤波处理
#             HoughTrans(binary_, src, result_path + file, h, w)  # 霍夫变换矫正
#     print("耗时：", time.time() - start)
