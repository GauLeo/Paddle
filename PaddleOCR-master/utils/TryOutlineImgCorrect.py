import cv2
import imageio
import numpy as np
import cv2

class OutlineCorrectUtil:  # 图片纠偏工具类
    @classmethod
    # 获取宽高
    def rotate_bound(image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # 提取旋转矩阵 sin cos
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # 计算图像的新边界尺寸
        nW = int((h * sin) + (w * cos))
        #     nH = int((h * cos) + (w * sin))
        nH = h

        # 调整旋转矩阵
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


## 获取图片旋转角度
    @classmethod
    def get_minAreaRect(cls,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
          angle = -(angle + 90)
        else:
          angle = -angle
        return angle

    # def getAngle(self,image):
    #     angle = get_minAreaRect(image)[-1]
    #     return get_minAreaRect(image)[-1]
