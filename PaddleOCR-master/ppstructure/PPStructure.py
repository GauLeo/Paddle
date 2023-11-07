
from ppstructure.predict_table import TableSystem,to_excel
from utility import init_args
import cv2
import copy
import numpy as np
import os
args = init_args().parse_args(args=[])
args.det_model_dir='../ch_PP-OCRv3_det_infer'
args.cls_model_dir='../ch_ppocr_mobile_v2.0_cls_infer'
args.rec_model_dir='../ch_PP-OCRv3_rec_infer'
args.use_angle_cls = 'true'
args.table_model_dir= './inference/en_ppocr_mobile_v2.0_table_structure_infer'
args.image_dir = '../img_correct/out/out11.png'
args.rec_char_dict_path = '../ppocr/utils/ppocr_keys_v1.txt'
args.table_max_len = 512
args.det_limit_type = 'max'
args.det_limit_side_len = 2500
args.det_db_unclip_ratio = 1.5
args.table_char_dict_path = '../ppocr/utils/dict/table_structure_dict.txt'
args.output='../output/table'
args.vis_font_path='../doc/fonts/simfang.ttf'
args.use_gpu=False


# 初始化表格识别系统
if __name__ == '__main__':
  table_sys = TableSystem(args)
  img = cv2.imread('../output/cut_1.png')
  pred_html = table_sys(img)
  print(pred_html)
  print("toexcel")
   # 结果存储到excel文件
  to_excel(pred_html,'1.xlsx')
  print(pred_html)