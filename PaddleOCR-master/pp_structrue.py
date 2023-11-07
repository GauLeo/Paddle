from paddleocr import PPStructure,draw_structure_result,save_structure_res
import cv2

img_path = './doc/imgs/test.jpg'
table_engine = PPStructure(show_log=True)
img = cv2.imread(img_path)
result = table_engine(img)
print(result)
save_structure_res(result,'aa','c')