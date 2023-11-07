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

from deploy.hubserving.structure_table.params import read_params as table_read_params


def read_params():
    cfg = table_read_params()
    cfg.layout_path_model = 'lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config'
    cfg.layout_label_map = None
    cfg.mode = 'structure'
    cfg.table_max_len = 512
    cfg.table_model_dir = './inference/en_ppocr_mobile_v2.0_table_structure_infer'
    cfg.table_char_dict_path = './ppocr/utils/dict/table_structure_dict.txt'
    cfg.rec_char_dict_path = './ppocr/utils/ppocr_keys_v1.txt'
    cfg.det_limit_side_len = 2500.0
    cfg.det_db_unclip_ratio = 1.5
    cfg.output = './output/table'
    cfg.vis_font_path = './doc/fonts/simfang.ttf'

    cfg.cls_batch_num = 6
    cfg.det_db_box_thresh = 0.6
    cfg.drop_score = 0.5
    cfg.rec_algorithm = 'SVTR_LCNet'
    cfg.show_log = True
    cfg.use_gpu = True
    return cfg
