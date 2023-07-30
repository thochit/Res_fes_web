import numpy as np
import cv2
import cv2 as cv
import os
import subprocess
import glob
from app.model3.options.test_options import TestOptions
from app.model3.model.net import InpaintingModel_DFBM
from app.model3.util.utils import generate_rect_mask
from torchvision import transforms
from time import sleep

# path temp files
path_pre_image = r"app/model3/temp0.png"
path_hsv_result = r"app/model3/temp1.png"

def m3_preprocessing_image(path:str):
    image = cv.imread(path)
    image = cv.resize(image, (256, 256))
    image = cv.GaussianBlur(image, (7, 7), 0)
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv2.imwrite(path_pre_image, image)    
    return image

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2) / (2 * sigma**2)), (size, size))
    return kernel

def convert_image_org(image):
    kernel = gaussian_kernel(7, 1.0)
    reverse_kernel = np.flipud(np.fliplr(kernel))
    image = cv.filter2D(image, -1, reverse_kernel)
    image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
    return image

def load_image(path:str):
  m3_preprocessing_image(path)
  image = cv2.imread(path_pre_image)
  image = np.transpose(image, [2, 0, 1])
  image = np.expand_dims(image, axis=0)
  return image

def predict(model, image, mask):
  result = model.evaluate(image, mask)
  result = np.transpose(result[0][::-1, :, :], [1, 2, 0])
  result = result[:, :, ::-1]
  # result = convert_image_org(result)
  cv2.imwrite(path_hsv_result, result)

def convert_final_result(path):
  image = cv2.imread(path_hsv_result)
  image = convert_image_org(image)
  cv2.imwrite(path, image)



path_model = r"app/model3/50_net_DFBN.pth"
# path_org_image = r"app/model3/image_107.PNG"
path_result = r"/media/thochit/DATA/PythonProject/Resfes/app/static/images/output3.jpg"

def model3(path):
  # load model
  config = config = TestOptions().parse()
  ourModel = InpaintingModel_DFBM(opt=config)
  ourModel.load_networks(path_model)

  # generate mask & image
  mask, _ = generate_rect_mask(config.img_shapes, config.mask_shapes, config.random_mask)
  image = load_image(path)

  # predict
  predict(ourModel, image, mask)
  convert_final_result(path_result)

# main()
