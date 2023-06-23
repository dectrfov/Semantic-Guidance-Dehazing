# CUDA_VISIBLE_DEVICES=0 python test.py
import os
import random
import math
import time
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import argparse

import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.parallel
import torchvision
print(torch.cuda.device_count())

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)

##############################################
# Model Part
##############################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_network(cfg):
    from model import fusion_net_base
    model = fusion_net_base()
    model = model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(cfg.test_weight)['state_dict'])
    return model.cuda()

##############################################
# Data Part
##############################################
def trans_img(img_path):
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    ori_image = data_transform(Image.open(str(img_path))).unsqueeze(0)
    ori_image = ori_image[:,:3,:,:]
    b, c, h, w = ori_image.shape
    return ori_image

##############################################
# Main Part
##############################################
def main(cfg):
    # Build the record files
    print(os.path.join(cfg.output_dir, cfg.exp_name))
    out_folder = os.path.join(cfg.output_dir, cfg.exp_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    # model
    network = load_network(cfg).cuda()
    network.eval()
    # data
    files = os.listdir(cfg.test_data_path)
    # start test
    print('Start test')
    t1 = time.time()
    for file_name in tqdm(files):
        with torch.no_grad():
            img = trans_img(os.path.join(cfg.test_data_path, file_name))

            img = img.cuda()
            out = network(img).cpu()
            img = img.cpu()

            img_t = torch.transpose(img, 2, 3).cuda()
            out_t = network(img_t).cpu()
            out_t = torch.transpose(out_t, 2, 3)
            img_t = img_t.cpu()

            img_f = torch.flip(img, (3,)).cuda()
            out_f = network(img_f).cpu()
            img_f = img_f.cpu()
            out_f = torch.flip(out_f, (3,))

            img_ft = torch.transpose(img_f, 2, 3).cuda()
            out_ft = network(img_ft).cpu()
            out_ft = torch.transpose(out_ft, 2, 3)
            out_ft = torch.flip(out_ft, (3,))

            img_fd = torch.flip(img_f, (2,)).cuda()
            out_fd = network(img_fd).cpu()
            out_fd = torch.flip(out_fd, (3,))
            out_fd = torch.flip(out_fd, (2,))

            img_fdt = torch.transpose(img_fd, 2, 3).cuda()
            out_fdt = network(img_fdt).cpu()
            out_fdt = torch.transpose(out_fdt, 2, 3)
            out_fdt = torch.flip(out_fdt, (2,))
            out_fdt = torch.flip(out_fdt, (3,))

            img_d = torch.flip(img, (2,)).cuda()
            out_d = network(img_d).cpu()
            out_d = torch.flip(out_d, (2,))

            img_dt = torch.transpose(img_d, 2, 3).cuda()
            out_dt = network(img_dt).cpu()
            out_dt = torch.transpose(out_dt, 2, 3)
            out_dt = torch.flip(out_dt, (2,))
            img_dt = img_dt.cpu()

            out = (out + out_f + out_fd + out_d + out_t + out_ft + out_fdt + out_dt) / 8.0
            b, c, h, w = img.shape

        torchvision.utils.save_image(out, os.path.join(out_folder, "1"+file_name))
        img = cv2.imread(os.path.join(out_folder, "1"+file_name))
        data = np.zeros((h, w, 4), dtype=np.uint8) + 255
        data[:,:,:3] = img
        cv2.imwrite(os.path.join(out_folder, file_name).replace('JPG', 'png'), data)
        os.system("rm {}".format(os.path.join(out_folder, "1"+file_name)))
    t2 = time.time()
    print("Time:", t2-t1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, default='../tests',  help='Validation haze image path')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--exp_name',  default='NTIRE23_Test_Best', type=str)
    parser.add_argument('--test_weight', type=str, default='../weights/best.pkl')

    config_args, unparsed_args = parser.parse_known_args()
    main(config_args)

