import os
import torch
import cv2 as cv
import numpy as np
import torch.nn as nn
from app.networks import Generator
from torchvision import transforms
from flask import Flask, render_template, request, redirect

def create_dict(input_dim: int = 3, ngf: int = 32, cuda: str = "cuda:0", cuda_ids: list = [0]):
    dict_model = {"netG": {"input_dim": input_dim, "ngf": ngf}, "cuda": cuda, "cuda_ids": cuda_ids}
    return dict_model


def preprocessing_image(path: str):
    image = cv.imread(path)
    image = cv.resize(image, (256, 256))
    image = cv.GaussianBlur(image, (7, 7), 0)
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image = transforms.ToTensor()(image)
    return image


def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)), (size, size))
    return kernel


def convert_image_org(image):
    kernel = gaussian_kernel(7, 1.0)
    reverse_kernel = np.flipud(np.fliplr(kernel))
    image = cv.filter2D(image, -1, reverse_kernel)
    image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
    return image


def random_bbox(batch_size):
    img_height, img_width, _ = 256, 256, 3
    h, w = 128, 128
    margin_height, margin_width = [0, 0]
    maxt = img_height - margin_height - h
    maxl = img_width - margin_width - w
    bbox_list = []
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)
    bbox_list.append((t, l, h, w))
    bbox_list = bbox_list * batch_size
    return torch.tensor(bbox_list, dtype=torch.int64)


def bbox2mask(bboxes, height, width, max_delta_h, max_delta_w):
    batch_size = bboxes.size(0)
    mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
    for i in range(batch_size):
        bbox = bboxes[i]
        delta_h = np.random.randint(max_delta_h // 2 + 1)
        delta_w = np.random.randint(max_delta_w // 2 + 1)
        mask[i, :, bbox[0] + delta_h:bbox[0] + bbox[2] - delta_h, bbox[1] + delta_w:bbox[1] + bbox[3] - delta_w] = 1.
    return mask

def normalize(x):
    return x.mul_(2).add_(-1)

def mask_image(x, bboxes):
    height, width, _ = 256, 256, 3
    max_delta_h, max_delta_w = 32, 32
    mask = bbox2mask(bboxes, height, width, max_delta_h, max_delta_w)
    result = x * (1. - mask)
    return result, mask

def create_mask(x, batch_size):
    boxxes = random_bbox(batch_size)
    ground_truth = normalize(x)
    ground_truth = ground_truth.unsqueeze(dim=0)
    x, mask = mask_image(ground_truth, boxxes)
    return x, mask


class Model(nn.Module):

    def __init__(self, dict):
        super(Model, self).__init__()

        self.dict = dict
        self.device_cuda = self.dict["cuda"]
        self.device_ids = self.dict["cuda_ids"]
        self.netG = Generator(self.dict["netG"], self.device_cuda, self.device_ids)

    def load_weight(self, path):
        model = self.netG
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return model


def tensor_to_numpy(x):
    result = x.cpu()
    result = result[0].permute(1, 2, 0)
    result = result.detach().numpy()
    return result




# # Load model;
# dict_model = create_dict()
# model = Model(dict_model)
# model = model.load_weight("gen_00065000.pt")
# model = model.cuda()
# model.eval()

# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/', methods=['POST'])
# def upload_image():
#     # Input image;
#     image1 = request.files['image']
#     image1.save("static/images/original.jpg")
#     image2 = "static/images/original.jpg"
#     image = preprocessing_image(image2)

#     # Create mask and predict;
#     x, mask = create_mask(image, 1)
#     x = x.cuda()
#     mask = mask.cuda()
#     x1, x2, offset_flow = model(x, mask)
#     x2 = x2.cuda()
#     inpainted_result = x2 * mask + x * (1. - mask)
#     inpainted_result = tensor_to_numpy(inpainted_result)

#     # Convert tensor to numpy;
#     x = tensor_to_numpy(x)
#     image = image.permute(1, 2, 0)
#     # image = image.numpy()

#     path_save = r"static/images"
#     # Path file;
#     # image_org = os.path.join(path_save, "original.jpg")
#     image_mask = os.path.join(path_save, "masked.jpg")
#     image_res = os.path.join(path_save, "output.jpg")
#     # Save image;
#     # cv.imwrite(filename=image_org, img=image * 255)
#     cv.imwrite(filename=image_mask, img=x * 255)
#     cv.imwrite(filename=image_res, img=inpainted_result * 255)

#     return redirect('/result')


# @app.route('/result')
# def result():
#     return render_template('result.html')

