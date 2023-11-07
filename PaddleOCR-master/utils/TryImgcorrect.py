import imageio
import numpy as np
import cv2
from scipy import ndimage
from skimage import io
import math
from utils import TryHolfImgCorrect
from utils import TryOutlineImgCorrect
class ImgCorrectUtil:  # 图片纠偏工具类
    @classmethod
    def delWhite(self,path):
        img = cv2.imread(path)
        img = img[:-6,: -6]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = 255 * (gray < 128).astype(np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
        coords = cv2.findNonZero(gray)
        x, y, w, h = cv2.boundingRect(coords)
        rect = img[y:y + h, x:x + w]
        cv2.imshow("Cropped", rect)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("Output.png", rect)

    @classmethod
    def shape_correction(cls,path,filename):
        img = cv2.imread(path)
        (height, width) = img.shape[:2]
        print(img.shape)

        img_gau = cv2.GaussianBlur(img, (5, 5), 0)
        canny = cv2.Canny(img_gau, 60, 200)
        # cv.imshow("g-canny", canny)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 3))

        dilated = cv2.dilate(canny, kernel, iterations=8)
        # cv.imshow('img_dilated', dilated)

        # 寻找轮廓

        # contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _,contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(len(contours), hierarchy, sep='\n')

        # 找到最外层面积最大的轮廓

        area = 0
        # print("area:{}".format(area))

        index = 0
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            # 排除非文本区域
            if w < 35 and h < 35:
                continue
            # 防止矩形区域过大不精准
            if h > 0.99 * height or w > 0.99 * width:
                continue
            # draw rectangle around contour on original image
            # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            tmpArea = w * h
            if tmpArea >= area:
                area = tmpArea
                index = i

        # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        rect = cv2.minAreaRect(contours[index])
        # 画出矩形框
        # box = cv.boxPoints(rect)
        # box = np.int0(box)
        # cv.drawContours(img, [box], 0, (0, 0, 255), 2)

        # cv.imshow('img', img)
        print("rect:{}".format(rect))
        angle = rect[-1]
        # print(angle)

        # 角度大于85度或小于5度不矫正
        if angle > 85 or angle < 5:
            angle = 0
        elif angle < 45:
            angle = angle - 0
        else:
            angle = angle - 90

        M = cv2.getRotationMatrix2D(rect[0], angle, 1)
        rotated = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
        cv2.putText(rotated, "angle: {:.2f} ".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        imageio.imwrite("./img/out/" + filename, rotated)
        # return rotated
        cv2.imshow("output", rotated)
        cv2.waitKey(0)

    @classmethod
    def correntImg(self, path,filename):
        image_path = path
        image = cv2.imread(image_path)
        outlineAngle = TryOutlineImgCorrect.OutlineCorrectUtil.get_minAreaRect(image)[-1]
        houghAngle = TryHolfImgCorrect.HolfCorrectUtil.calcDegree( image_path)
        angle = outlineAngle
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # 提取旋转矩阵 sin cos
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # 计算图像的新边界尺寸
        nW = int((h * sin) + (w * cos))
        #     nH = int((h * cos) + (w * sin))
        nH = h

        # 调整旋转矩阵
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        rotated = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.putText(rotated, "angle: {:.2f} ".format(angle),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        imageio.imwrite("./img/out/"+filename, rotated)
        cv2.imshow("imput", image)
        cv2.imshow("output", rotated)
        cv2.waitKey(0)


