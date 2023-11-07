#矫正倾斜图片
import os
import cv2
import math
import numpy as np
from scipy import ndimage
import imageio

# 源文件夹，注意不能有中文
filepath = './img_correct'

# 遍历文件
for filename in os.listdir(filepath):
    # 读取图像
    img = cv2.imread(filepath + '/%s' % filename)
    # 二值化
    if img is None:
        break
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 边缘检测
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    #霍夫变换，摘自https://blog.csdn.net/feilong_csdn/article/details/81586322
    lines = cv2.HoughLines(edges,1,np.pi/180,0)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    if x1 == x2 or y1 == y2:
        continue
    t = float(y2-y1)/(x2-x1)
    # 得到角度后
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    # 图像根据角度进行校正
    rotate_img = ndimage.rotate(img, rotate_angle)

    # 输出图像
    imageio.imwrite(filepath + '/out/' + filename, rotate_img)
    print(filename + " 已转换")
print("转换结束")