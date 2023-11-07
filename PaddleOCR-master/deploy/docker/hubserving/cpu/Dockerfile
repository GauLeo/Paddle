# Version: 2.3.0

FROM registry.baidubce.com/paddlepaddle/paddle:2.3.0

# PaddleOCR base on Python3.8
RUN pip3 install --upgrade pip -i https://mirror.baidu.com/pypi/simple

RUN pip3 install paddlehub --upgrade -i https://mirror.baidu.com/pypi/simple

RUN git clone https://github.com/ZQSGSZD/PaddleOCR.git /PaddleOCR

WORKDIR /PaddleOCR

RUN pip3 install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

RUN pip3 install -U https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl

# RUN pip3 install commands -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

RUN mkdir -p /PaddleOCR/inference/
# Download orc detect model(light version). if you want to change normal version, you can change ch_ppocr_mobile_v2.0_det_infer to ch_ppocr_server_v2.0_det_infer, also remember change det_model_dir in deploy/hubserving/ocr_system/params.py）

ADD https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/ch_PP-OCRv3_det_infer.tar  -C /PaddleOCR/inference/

# Download direction classifier(light version). If you want to change normal version, you can change ch_ppocr_mobile_v2.0_cls_infer to ch_ppocr_mobile_v2.0_cls_infer, also remember change cls_model_dir in deploy/hubserving/ocr_system/params.py）

ADD https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/ch_ppocr_mobile_v2.0_cls_infer.tar -C /PaddleOCR/inference/

# Download orc recognition model(light version). If you want to change normal version, you can change ch_ppocr_mobile_v2.0_rec_infer to ch_ppocr_server_v2.0_rec_infer, also remember change rec_model_dir in deploy/hubserving/ocr_system/params.py）
ADD https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/ch_PP-OCRv3_rec_infer.tar -C /PaddleOCR/inference/



ADD https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/en_ppocr_mobile_v2.0_table_structure_infer.tar  -C /PaddleOCR/inference/



EXPOSE 8868

CMD ["/bin/bash","-c","hub install deploy/hubserving/structure_system/ && hub serving start -m structure_system"]