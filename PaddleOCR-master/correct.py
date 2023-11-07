from utils import ImgCorrect
from utils import Batchidentify
import os
from utils import PerspectiveCorrection
imgCorrectUtil = ImgCorrect.ImgCorrectUtil
imgCorrectUtil.correctImg('./pdf_correct/pdf_out/images_0_1.jpg','./pdf_correct/pdf_out/images_0_1_correct.jpg')
# imgCorrectUtil.correctImg('./output/yangss.png','./output/yangss_out.png')
# perspectiveUtil = PerspectiveCorrection.testutil
# perspectiveUtil.corp_margin('./pdf_correct/img/images_0.png')
#
# pdfUtil = Batchidentify.BatchidentifyUtil
# pdfUtil.pyMuPDF_fitz('./pdf_correct/test1.pdf','./pdf_correct/pdf_out1')
# print('pdf finish')
# imgpath = './pdf_correct/pdf_out'
# outpath = './pdf_correct/img'
# for filename in os.listdir(imgpath):
#     imgCorrectUtil.correctImg(imgpath + '/%s' % filename,outpath+'/%s'%filename)