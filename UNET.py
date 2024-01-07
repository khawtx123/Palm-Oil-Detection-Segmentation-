import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask


class Unet():
    def __init__(self, model_path, in_files, out_files):
        self.model_path = model_path
        self.in_files = in_files
        self.out_files = out_files

    def predict_img(self, net,
                    full_img,
                    device,
                    scale_factor=1,
                    out_threshold=0.9):
        net.eval()
        img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = net(img).cpu()
            output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
            if net.n_classes > 1:
                mask = output.argmax(dim=1)
            else:
                mask = torch.sigmoid(output) > out_threshold

        return mask[0].long().squeeze().numpy()

    def mask_to_image(self, mask: np.ndarray, mask_values):
        if isinstance(mask_values[0], list):
            out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
        elif mask_values == [0, 1]:
            out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
        else:
            out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

        if mask.ndim == 3:
            mask = np.argmax(mask, axis=0)

        for i, v in enumerate(mask_values):
            out[mask == i] = v

        return Image.fromarray(out)

    def run(self, plot = False):
        net = UNet(n_channels=3, n_classes=2, bilinear=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        net.to(device=device)
        state_dict = torch.load(self.model_path, map_location=device)
        mask_values = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)

        logging.info('Model loaded!')

        for i, filename in enumerate(self.in_files):
            logging.info(f'Predicting image {filename} ...')
            img = Image.open(filename)

            mask = self.predict_img(net=net,
                               full_img=img,
                               scale_factor=0.5,
                               out_threshold=0.8,
                               device=device)

            out_filename = self.out_files[i]
            result = self.mask_to_image(mask, mask_values)
            result.save(out_filename)
            if plot:
                plot_img_and_mask(img, mask)

    def run(self, in_file, plot = False):
        in_files = [in_file]
        net = UNet(n_channels=3, n_classes=2, bilinear=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        net.to(device=device)
        state_dict = torch.load(self.model_path, map_location=device)
        mask_values = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)

        logging.info('Model loaded!')

        for i, filename in enumerate(in_files):
            logging.info(f'Predicting image {filename} ...')
            img = Image.open(filename)

            mask = self.predict_img(net=net,
                               full_img=img,
                               scale_factor=0.5,
                               out_threshold=0.9,
                               device=device)

            out_filename = self.out_files[i]
            result = self.mask_to_image(mask, mask_values)
            result.save(out_filename)
            if plot :
                plot_img_and_mask(img, mask)

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.9):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()



def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

if __name__ == '__main__':
    model_path = "model/checkpoint_epoch100.pth"
    in_files = ["detection/detected_frame.jpg"]
    out_files = ["detection/mask.jpg"]

    net = UNet(n_channels=3, n_classes=2, bilinear=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=0.5,
                           out_threshold=0.9,
                           device=device)

        out_filename = out_files[i]
        result = mask_to_image(mask, mask_values)
        result.save(out_filename)
        plot_img_and_mask(img, mask)
