# Annotation Tool - padocr_refine.py

This is a Python script that provides an OCR image annotation tool. The tool allows you to annotate text in images using Optical Character Recognition (OCR) and create logs and annotations for the detected text blocks.

## Prerequisites

Make sure you have the following libraries installed:

- `os`
- `filetype`
- `csv`
- `numpy`
- `pandas`
- `cv2` (OpenCV)
- `paddleocr`

You can install these libraries using pip:

```
pip install filetype pandas opencv-python paddlepaddle paddleocr
```

## Usage

1. Import the required libraries.
2. Initialize the variables.
3. Prepare the list of images to be passed through OCR by calling the `make_img_list` function. Provide the path to the images folder as an argument or enter it when prompted.
4. Create an OCR object by calling the `make_ocr` function. You can specify the language (default is English).
5. Use the `make_logs` function to create logs and simple annotations for the detected text blocks in the images. Provide the images folder path, the image list, and the OCR object as arguments. The function returns the log file names, annotated images, and image names.
6. Use the `add_logs` function to manually add missed text blocks to the annotations. Provide the images folder path, log file names, and annotated images as arguments. The function allows you to draw bounding boxes around the text blocks and enter the corresponding text for annotation. The function returns the updated annotated images and dataframes.
7. Use the `corr_logs` function to correct any incorrect labelings in the annotations. Provide the images folder path, image names, annotated images, and dataframes as arguments. The function allows you to select and modify existing bounding boxes and texts. The corrected images and dataframes are saved in a separate directory.
8. Execute the main code by calling the necessary functions in the correct order.

## Example

Here is an example of how to use the OCR image annotation tool:

```python
# Import required libraries
import os
import filetype as ft
import csv
import numpy as np
import pandas as pd
import cv2
from paddleocr import PaddleOCR

# The rest of the code...

# Execution
if __name__ == "__main__":
    im_list, dir_path = make_img_list()
    main_ocr = make_ocr()
    basic_csvs, basic_anns, img_names = make_logs(dir_path, im_list, main_ocr)
    new_anns, new_dfs = add_logs(dir_path, basic_csvs, basic_anns)
    corr_logs(dir_path, img_names, new_anns, new_dfs)
```

Make sure to replace the paths and customize the code according to your needs.

## License

This code is released under the [MIT License](https://opensource.org/licenses/MIT).
