import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import TryHolfTrans
import math
#
# # 完成灰度化，二值化
# def two_value(img_raw):
#     img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)  # 灰度化
#     ret, img_two = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 二值化
#     return img_two
#
#
# # 旋转函数
# def rotate(img_rotate_raw, angle):
#     print(angle)
#     (h, w) = img_rotate_raw.shape[:2]
#     (cx, cy) = (w // 2, h // 2)
#     m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)  # 计算二维旋转的仿射变换矩阵
#     return cv2.warpAffine(img_rotate_raw, m, (w, h), borderValue=(0, 0, 0))
#
#
# # 霍夫直线检测
# def get_angle(img_hou_raw):
#     sum_theta = 0
#     img_canny = cv2.Canny(img_hou_raw, 50, 200, 3)
#     lines = cv2.HoughLines(img_canny, 1, np.pi / 180, 300, 0, 0)
#     # lines 是三维的
#     for i in range(lines.shape[0]):
#         theta = lines[i][0][1]
#         sum_theta += theta
#     average = sum_theta / lines.shape[0]
#     angle = average / np.pi * 180 - 90
#     return angle
#
#
# def correct(img_cor_raw):
#     img_two = two_value(img_cor_raw)
#     angle = get_angle(img_two)
#     if angle == -1:
#         print("No lines!!!")
#         return 0
#     return rotate(img_two, angle)
#
#
# if __name__ == "__main__":
#     image_path = '../img/1_1.png'
#     image = cv2.imread(image_path)
#     cv2.imshow("raw", image)
#     img_rot2 = correct(image)
#     cv2.imshow("last", img_rot2)
#     cv2.waitKey()


class HolfCorrectUtil:  # 图片纠偏工具类
    def binary(img):
        '''
        :param img: 原始图像
        :return: 二值化后的图像
        '''
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cvtColor用于转换色彩空间，将图像转换为GRAY灰度图像
        #     gray = cv.medianBlur(gray,5)  # 中值滤波
        gray = cv2.GaussianBlur(gray, (9, 9), 9)  # 高斯滤波
        ret, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # 由于文本是黑底白字的，需要让背景是黑色的，所以在传入参数时需要使用cv.THRESH_BINARY_INV 加上_INV使二值图反转
        return binary

    @classmethod
    def calcDegree(cls,filepath):
      # src = cv2.imread(filepath)
      # src = cv2.copyMakeBorder(src, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])
      # (h, w) = src.shape[:2]
      # gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # cvtColor用于转换色彩空间，将图像转换为GRAY灰度图像
      # #     gray = cv.medianBlur(gray,5)  # 中值滤波
      # gray = cv2.GaussianBlur(gray, (9, 9), 9)  # 高斯滤波
      # ret, binary = cv2.threshold(gray, 0, 255,
      #                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
      #  # 二值化、滤波处理
      # HolfTrans.HoughTransUtil.HoughTrans(binary, src, './img/houghcorrect/result' , h, w)  # 霍夫变换矫正

      # 通过霍夫变换计算角度
      img_path = filepath
      srcImage = cv2.imread(img_path)
      midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
      dstImage = cv2.Canny(midImage, 50, 200, 3)
      lineimage = srcImage.copy()
      # 通过霍夫变换检测直线
      # 第4个参数就是阈值，阈值越大，检测精度越高
      lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 0)
      # 由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
      sum = 0
      # 依次画出每条线段

      for i in range(len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(round(x0 + 1000 * (-b)))
            y1 = int(round(y0 + 1000 * a))
            x2 = int(round(x0 - 1000 * (-b)))
            y2 = int(round(y0 - 1000 * a))

            # 只选角度最小的作为旋转角度
            sum += theta
            cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("holfLine", lineimage)


      # 对所有角度求平均，这样做旋转效果会更好
      average = sum / len(lines)
      degreeTrans = average / np.pi * 180
      angle = -(degreeTrans - 90)
      print(angle)
      return angle

