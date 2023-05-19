# import required libraries
import os
import filetype as ft
import csv
import numpy as np
import cv2
from paddleocr import PaddleOCR

# prepare list of images to be passed through OCR
def make_img_list(dir_path=None):
    if dir_path is None:
        dir_path = input("Enter images folder path: ")
    img_list = []
    for path in os.listdir(dir_path):
        path_full = os.path.join(dir_path, path)
        if ft.is_image(path_full):
            img_list.append(path_full)
    return img_list, dir_path

# Create OCR object, default English
def make_ocr(language='en'):
    # make it versatile for angled text
    ocr = PaddleOCR(use_angle_cis=True, lang=language)
    return ocr

# To see information stoarge mannerisms
def extract_text(ocr_output):
    for sublist in ocr_output:
        boxes = [item[0] for item in sublist]
        txts = [item[1][0] for item in sublist]
        scores = [item[1][1] for item in sublist]
        return boxes, txts, scores

# Make logs of all images in the list via OCR
def make_logs(dir_path, img_list, ocr):
    for img in img_list:
        result = ocr.ocr(img, cls=True)
        boxes, txts, scores = extract_text(result)
        img_name = (str(img).split('/')[-1]).split('.')[0]
        log_name = dir_path + '/' + img_name + '_logs.csv'
        with open(log_name, 'w') as f:
            csvwriter = csv.writer(f, quoting=csv.QUOTE_ALL)
            for i in range(len(boxes)):
                csvwriter.writerow([boxes[i], txts[i], scores[i]])

# Execution
if __name__ == "__main__":
    im_list, dir_path = make_img_list()
    main_ocr = make_ocr()
    make_logs(dir_path, im_list, main_ocr)
