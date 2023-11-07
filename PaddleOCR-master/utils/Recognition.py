import cv2
from matplotlib import pyplot as plt

class DistinguishUtil():
    @classmethod
    def extract_text(cls,imgpath):
        img = cv2.imread(imgpath)
