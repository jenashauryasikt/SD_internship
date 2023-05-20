# import required libraries
import os
import filetype as ft
import csv
import numpy as np
import pandas as pd
import cv2
from paddleocr import PaddleOCR

# Initialize variables
annotations = []
drawing = False
bbox = []
text = ""
previous_edit = None
zoom_scale = 1.0
zoom_center = None
panning_offset = [0, 0]

# prepare list of images to be passed through OCR
def make_img_list(dir_path=None):
    if dir_path is None:
        dir_path = input("Enter images folder path: ")
    img_list = []
    for path in os.listdir(dir_path):
        path_full = os.path.join(dir_path, path)
        if os.path.isfile(path_full):
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
    # create directory for image logs - basic
    logs_path = os.path.join(dir_path, 'logs')
    if not (os.path.exists(logs_path)):
        os.mkdir(logs_path)
    # create directory for annotations - basic
    ann_path = os.path.join(dir_path, 'ann')
    if not (os.path.exists(ann_path)):
        os.mkdir(ann_path)        
    log_names = []
    images = []
    img_names = []
    for img in img_list:
        image = cv2.imread(img)
        result = ocr.ocr(img, cls=True)
        # obtain bounding boxes of text blocks with confidence scores
        boxes, txts, scores = extract_text(result)
        # identify file name without any extension
        img_name = (str(img).split('/')[-1]).split('.')[0]
        img_names.append(img_name)
        # store logs and simple annotations
        log_name = logs_path + '/' + img_name + '_logs.csv'
        new_img_name = ann_path + '/' + img_name + '_ann.jpg'
        with open(log_name, 'w') as f:
            csvwriter = csv.writer(f, quoting=csv.QUOTE_ALL)
            for i in range(len(boxes)):
                csvwriter.writerow([boxes[i], txts[i], scores[i]])
                # annotate already detected boxes
                bbox = np.array(boxes[i]) + panning_offset
                text = txts[i]
                image = cv2.rectangle(image, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 255, 0), 2)
                image = cv2.putText(image, text, (int(bbox[0][0]), int(bbox[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite(new_img_name, image)
        log_names.append(log_name)
        images.append(image)
    return log_names, images, img_names
    
# # Add missed text blocks
def add_logs(dir_path, log_names, images):
    # Create a directory for new logs
    new_logs_path = os.path.join(dir_path, 'corr_logs')
    if not os.path.exists(new_logs_path):
        os.mkdir(new_logs_path)
    # Create a directory for new annotations
    new_ann_path = os.path.join(dir_path, 'corr_ann')
    if not os.path.exists(new_ann_path):
        os.mkdir(new_ann_path)
    new_log_dfs = []
    new_images = []
    for i in range(len(images)):
        image = images[i]
        log_name = log_names[i]
        log_df = pd.read_csv(log_name, header=None)
        # Create a copy of the image to draw on
        clone = image.copy()
        # Initialize variables for bounding box drawing
        drawing = False
        bbox_start = (0, 0)
        bbox_end = (0, 0)
        def draw_bbox(event, x, y, flags, param):
            nonlocal drawing, bbox_start, bbox_end, log_df
            if event == cv2.EVENT_LBUTTONDOWN:
                # Left mouse button down, start drawing the bounding box
                drawing = True
                bbox_start = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                # Left mouse button released, stop drawing the bounding box
                drawing = False
                bbox_end = (x, y)
                cv2.rectangle(clone, bbox_start, bbox_end, (255, 0, 0), 2)
                cv2.imshow('Image', clone)
                # Prompt the user to enter the text for annotation
                text = "" # Clear the variable
                text = input("Enter the text: ")
                cv2.putText(clone, text, (bbox_start[0], bbox_start[1] - 10),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                # Add the drawn bounding box and text to the log
                x1, y1 = bbox_start
                x2, y2 = bbox_end                
                # Obtain the coordinates of the bounding box corners
                bbox_corners = str([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                log_df.loc[len(log_df)] = [bbox_corners, text, np.nan]
                # log_df = log_df.append(pd.Series([bbox_corners, text], index=log_df.columns), ignore_index=True)
        # Create a new window to display the image
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Image', draw_bbox)
        # Get the system monitor size
        monitor_width = cv2.getWindowImageRect('Image')[2]
        monitor_height = cv2.getWindowImageRect('Image')[3]
        # Resize the window to match the system monitor size
        cv2.resizeWindow('Image', monitor_width, monitor_height)
        while True:
            cv2.imshow('Image', clone)
            # Handle keyboard events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Quit the loop when 'q' is pressed
                break
            elif key == ord('r'):
                # Remove the drawn bounding box and text if 'r' is pressed
                clone = image.copy()
                log_df = pd.read_csv(log_name, header=None)
        new_images.append(clone)
        new_log_dfs.append(log_df)
        # Close the image window
        cv2.destroyAllWindows()    
    return new_images, new_log_dfs

## Correct incorrect labellings
def corr_logs(dir_path, img_names, images, log_dfs):
    # Create a directory for new corrected logs
    corr_logs_path = os.path.join(dir_path, 'corr_logs')
    if not os.path.exists(corr_logs_path):
        os.mkdir(corr_logs_path)
    # Create a directory for new corrected annotations
    corr_ann_path = os.path.join(dir_path, 'corr_ann')
    if not os.path.exists(corr_ann_path):
        os.mkdir(corr_ann_path)
    corr_log_dfs = []
    corr_images = []
    for i in range(len(images)):
        image = images[i]
        log_df = log_dfs[i]
        img_name = img_names[i]
        # Create a copy of the image to draw on
        clone = image.copy()
        # Retrieve bounding boxes and text from the dataframe
        boxes = log_df[log_df.columns[0]].values.tolist()
        texts = log_df[log_df.columns[1]].values.tolist()
        # Initialize variables for drawing and annotation
        drawing = False
        selected_box = None
        def draw_bboxes(event, x, y, flags, param):
            nonlocal drawing, selected_box, x1, y1, x2, y2, j
            if event == cv2.EVENT_LBUTTONDOWN:
                # Left mouse button down, check if the click is inside any bounding box
                for j, box in enumerate(boxes):
                    a1, a2, a3, a4 = eval(box)
                    x1, y1, x2, y2 = int(a1[0]), int(a1[1]), int(a3[0]), int(a3[1])
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        selected_box = j
                        drawing = True
                        break
            elif event == cv2.EVENT_LBUTTONUP:
                # Left mouse button released, stop drawing
                drawing = False
                color = (0, 0, 255)
                cv2.rectangle(clone, (x1, y1), (x2, y2), color, 2)
                cv2.putText(clone, texts[j], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                # selected_box = None

        # Create a new window to display the image
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Image', draw_bboxes)
        # Get the system monitor size
        monitor_width = cv2.getWindowImageRect('Image')[2]
        monitor_height = cv2.getWindowImageRect('Image')[3]
        # Resize the window to match the system monitor size
        cv2.resizeWindow('Image', monitor_width, monitor_height)        
        while True:
            for j, box in enumerate(boxes):
                color = (0, 255, 0)  # Default color (green) for unselected bounding boxes
                if j == selected_box:
                    color = (0, 0, 255)  # Selected bounding box turns red
                    a1, a2, a3, a4 = eval(box)
                    x1, y1, x2, y2 = int(a1[0]), int(a1[1]), int(a3[0]), int(a3[1])
                    cv2.rectangle(clone, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(clone, texts[j], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow('Image', clone)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Quit the loop when 'q' is pressed
                break
            elif key == 13 and selected_box is not None:
                # Prompt the user to enter a new annotation for the selected bounding box
                text = input("Enter the new annotation: ")
                log_df.iloc[selected_box, 1] = text
        # Save the corrected image and dataframe
        corr_image_name = os.path.join(corr_ann_path, str(img_name)+'_corr.jpg')
        corr_log_name = os.path.join(corr_logs_path, str(img_name)+'_corr.csv')
        cv2.imwrite(corr_image_name, clone)
        log_df.to_csv(corr_log_name, index=False)

# Execution
if __name__ == "__main__":
    im_list, dir_path = make_img_list()
    main_ocr = make_ocr()
    basic_csvs, basic_anns, img_names = make_logs(dir_path, im_list, main_ocr)
    new_anns, new_dfs = add_logs(dir_path, basic_csvs, basic_anns)
    corr_logs(dir_path, img_names, new_anns, new_dfs)