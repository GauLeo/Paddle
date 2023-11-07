# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.insert(0, ".")
import copy

import sys, os
import time
# from paddlehub.logger import logger
from paddlehub.module.module import moduleinfo, runnable, serving
from ppstructure.table.predict_table import TableSystem
import cv2
import numpy as np
import paddlehub as hub
from ppstructure.predict_system import to_excel

from tools.infer.utility import base64_to_cv2
from ppstructure.predict_system import StructureSystem as PPStructureSystem
from ppstructure.predict_system import save_structure_res
from ppstructure.utility import parse_args
from deploy.hubserving.structure_system.params import read_params


@moduleinfo(
    name="structure_system",
    version="1.0.0",
    summary="PP-Structure system service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/structure_system")
class StructureSystem(hub.Module):
    def _initialize(self, use_gpu=False, enable_mkldnn=False):
        """
        initialize with the necessary elements
        """
        cfg = self.merge_configs()

        cfg.use_gpu = use_gpu
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
                print("use gpu: ", use_gpu)
                print("CUDA_VISIBLE_DEVICES: ", _places)
                cfg.gpu_mem = 8000
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
        cfg.ir_optim = True
        cfg.enable_mkldnn = enable_mkldnn
        print("TableSystem(cfg)")
        print(cfg)
        self.table_sys = TableSystem(cfg)
        # self.table_sys = PPStructureSystem(cfg)

    def merge_configs(self):
        # deafult cfg
        backup_argv = copy.deepcopy(sys.argv)
        sys.argv = sys.argv[:1]
        cfg = parse_args()

        update_cfg_map = vars(read_params())

        for key in update_cfg_map:
            cfg.__setattr__(key, update_cfg_map[key])

        sys.argv = copy.deepcopy(backup_argv)
        return cfg

    # 图像纠偏
    def read_images(self, paths=[]):
        images = []
        for img_path in paths:
            assert os.path.isfile(
                img_path), "The {} isn't a valid file.".format(img_path)
            img = cv2.imread(img_path)
            if img is None:
                # logger.info("error in loading image:{}".format(img_path))
                continue
            images.append(img)
        return images

    # 预测
    def predict(self, images=[], pagenumber=0, output_path=""):
        print("predict")
        print(pagenumber)
        print(output_path)
        """
        Get the chinese texts in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
        Returns:
            res (list): The result of chinese texts and save path of images.
        """
        output = output_path
        if images != [] and isinstance(images, list):
            predicted_data = images
        else:
            raise TypeError("The input data is inconsistent with expectations.")
        # if images != [] and isinstance(images, list) and paths == []:
        #     predicted_data = images
        # elif images == [] and isinstance(paths, list) and paths != []:
        #     predicted_data = self.read_images(paths)
        # else:
        #     raise TypeError("The input data is inconsistent with expectations.")
        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."
        all_results = []
        result = []
        for img in predicted_data:
            if img is None:
                # logger.info("error in loading image")
                all_results.append([])
                continue
            img_name = 'img_' + str(pagenumber)
            img_path = os.path.join(output, '{}.jpg'.format(img_name))
            print(img_path)
            ori_img = img
            print(ori_img)
            cv2.imwrite(img_path, ori_img)
            # cv2.imencode('.jpg', img).tofile(img_path)
            print("生成图片")
            starttime = time.time()
            cv_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite('./output/correct.png', img)
            angle = StructureSystem.findangle(cv_img)
            img = StructureSystem.rotate_bound(cv_img, -angle)
            # cv2.putText(img, 'Angle:{:.2f} degrees'.format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
            #             2)
            # 绘制文字
            # cv2.imwrite('./hubserving_result/test.png', img)
            print("纠偏完成")

            result = self.table_sys(img)
            print("切换predict_table结果为")
            print(result)
            pred_html = result['html']
            if result['html'] is not None:
                print("生成html_table")
                file = open(output + "/" + img_name + ".txt", 'w')
                file.write(pred_html)
                print("生成识别excel")
                to_excel(pred_html, output + '/{}.xlsx'.format(img_name))  # htmltable

            # elapse = time.time() - starttime
            # logger.info("Predict time: {}".format(elapse))
            # # parse result
            # res_final = []
            # for region in res:
            #     region.pop('img')
            #     res_final.append(region)
            # all_results.append({'regions': res_final})
            # for region in res_final:
            #     if region['type'] == 'Table':
            #         html = region['res']['html']
            #         print("生成excel文件")
            #         to_excel(html, output+'/{}.xlsx'.format(img_name))  # htmltable
            #         file = open(output + "/" + img_name + ".txt", 'w')
            #         file.write(html)
            #     if region['type'] == 'Figure':
            #         print("Figure")
            #         # x1, y1, x2, y2 = region['bbox']
            #         print(region['bbox'])
            #         # roi_img = img[y1:y2, x1:x2, :]
            #         # img_path = os.path.join(output, '{}.jpg'.format(img_name))
            #         # cv2.imwrite(img_path, roi_img)
        print("result")
        print(result)
        return result





    # 图像纠偏
    def rotate_bound(image, angle):
        # 获取宽高
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        img = cv2.warpAffine(image, M, (w, h))
        return img

    # 图像纠偏
    def rotate_points(points, angle, cX, cY):
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0).astype(np.float16)
        a = M[:, :2]
        b = M[:, 2:]
        b = np.reshape(b, newshape=(1, 2))
        a = np.transpose(a)
        points = np.dot(points, a) + b
        points = points.astype(np.int_)
        return points

    # 图像纠偏
    def findangle(_image):
        # 用来寻找当前图片文本的旋转角度 在±90度之间
        # toWidth: 特征图大小：越小越快 但是效果会变差
        # minCenterDistance：每个连通区域坐上右下点的索引坐标与其质心的距离阈值 大于该阈值的区域被置0
        # angleThres：遍历角度 [-angleThres~angleThres]

        cv2.imwrite('./pdf_correct/pdf_out/test1.png', _image)
        toWidth = _image.shape[1] // 2  # 500
        minCenterDistance = toWidth / 20  # 10
        angleThres = 30 * 100

        image = _image.copy()
        h, w = image.shape[0:2]
        if w > h:
            maskW = toWidth
            maskH = int(toWidth / w * h)
        else:
            maskH = toWidth
            maskW = int(toWidth / h * w)
        # 使用黑色填充图片区域
        swapImage = cv2.resize(image, (maskW, maskH))
        grayImage = cv2.cvtColor(swapImage, cv2.COLOR_BGR2GRAY)
        gaussianBlurImage = cv2.GaussianBlur(grayImage, (3, 3), 0, 0)
        histImage = cv2.equalizeHist(~gaussianBlurImage)
        binaryImage = cv2.adaptiveThreshold(histImage, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
        # pointsNum: 遍历角度时计算的关键点数量 越多越慢 建议[5000,50000]之中
        pointsNum = np.sum(binaryImage != 0) // 2
        # # 使用最小外接矩形返回的角度作为旋转角度
        # # >>一步到位 不用遍历
        # # >>如果输入的图像切割不好 很容易受干扰返回0度
        ##element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        ##dilated = cv2.dilate(binaryImage*255, element)
        ##dilated = np.pad(dilated,((50,50),(50,50)),mode='constant')
        ##cv2.imshow('dilated', dilated)
        ##cv2.waitKey(0)
        ##cv2.destroyAllWindows()
        ##cv2.waitKey(1)
        ##coords = np.column_stack(np.where(dilated > 0))
        ##angle = cv2.minAreaRect(coords)
        ##print("外接矩形：",angle)

        # 使用连接组件寻找并删除边框线条
        # >>速度比霍夫变换快5~10倍 25ms左右
        # >>计算每个连通区域坐上右下点的索引坐标与其质心的距离，距离大的即为线条
        connectivity = 8
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImage, connectivity, cv2.CV_8U)
        labels = np.array(labels)
        maxnum = [(i, stats[i][-1], centroids[i]) for i in range(len(stats))]
        maxnum = sorted(maxnum, key=lambda s: s[1], reverse=True)
        if len(maxnum) <= 1:
            return 0
        for i, (label, count, centroid) in enumerate(maxnum[1:]):
            cood = np.array(np.where(labels == label))
            distance1 = np.linalg.norm(cood[:, 0] - centroid[::-1])
            distance2 = np.linalg.norm(cood[:, -1] - centroid[::-1])
            if distance1 > minCenterDistance or distance2 > minCenterDistance:
                binaryImage[labels == label] = 0
            else:
                break
        # cv2.imshow('after process', binaryImage*255)

        minRotate = 0.00
        minCount = -1.00
        (cX, cY) = (maskW // 2, maskH // 2)

        points = np.column_stack(np.where(binaryImage > 0))[:pointsNum].astype(np.int16)
        for rotate in range(-angleThres, angleThres):
            rotateAngle = 0.00
            rotateAngle = rotate / 100.00
            # print("rotateAngle:",rotateAngle)
            rotatePoints = StructureSystem.rotate_points(points, rotateAngle, cX, cY)
            rotatePoints = np.clip(rotatePoints[:, 0], 0, maskH - 1)
            hist, bins = np.histogram(rotatePoints, maskH, [0, maskH])
            # 横向统计非零元素个数 越少则说明姿态越正
            zeroCount = np.sum(hist > toWidth / 50)
            if zeroCount <= minCount or minCount == -1:
                minCount = zeroCount
                minRotate = rotateAngle

        print("over: rotate = ", minRotate)
        return minRotate

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        print("serving_method")
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.predict(images_decode, **kwargs)
        return results


if __name__ == '__main__':
    structure_system = StructureSystem()
    structure_system._initialize()
    image_path = ['./pdf_correct/pdf_out/images_0.png']
    res = structure_system.predict(paths=image_path,)
    print(res)
