"""
Created on Tue Oct 4 2022
@author: ynomura
"""

import argparse
from datetime import datetime as dt
import imageio
import numpy as np
import os
import sys
import torch

from overlay import *
import dice
from dataset_segmentation import DataSetSegmentation
from unet import Unet
from attUnet import attUnet
import scipy.ndimage as ndimg

def labeling(outputs, labels, binarize_threshold):
    out = outputs.cpu().numpy()
    out_mask = (out[0, 0, :, :] >= binarize_threshold).astype(np.uint8)
    label_img = labels.cpu().numpy()
    label_img = (label_img[0, 0, :, :] >= binarize_threshold).astype(np.uint8)
    
    # labeling 
    label = ndimg.label(out_mask)
    areas = np.array(ndimg.sum(out_mask, label[0], np.arange(label[0].max()+1)))
    mask = areas > (sum(areas) * 0.25)
    out_mask = mask[label[0]].astype(np.uint8)
    return out_mask, label_img

def test(test_data_path, model_file_name, output_path,
         binarize_threshold=0.5, gpu_id="0",
         export_mask=False, time_stamp="", **kwargs):

    if not os.path.isdir(test_data_path):
        print(f"Error: Path of test data ({test_data_path}) is not found.")
        sys.exit(1)

    # Check model file is exist
    if not os.path.isfile(model_file_name):
        print(f"Error: Model file ({model_file_name}) is not found.")
        sys.exit(1)

    # Automatic creation of output folder
    if not os.path.isdir(output_path):
        print(f"Path of output data ({output_path}) is created automatically.")
        os.makedirs(output_path)        

    # Set ID of CUDA device
    device = f'cuda:{gpu_id}'
    print(f"Device: {device}")

    # Create dataset
    test_dataset = DataSetSegmentation(test_data_path)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=True)

    # Define network (U-Net)
    in_channels = test_dataset[0][0].shape[0]
    out_channels = test_dataset[0][1].shape[0]
    first_filter_num = kwargs.get('first_filter_num', 16)
    if model == 0:
        model = Unet(in_channels, out_channels, first_filter_num)
    elif model == 1:
        model = attUnet(in_channels, out_channels, first_filter_num)

    # Load the parameter of model
    model.load_state_dict(torch.load(model_file_name, map_location=device))
    model = model.to(device)
    model.eval()

    if time_stamp == "":
        time_stamp = dt.now().strftime('%Y%m%d%H%M%S')

    result_file_name = f'{output_path}/test_result_th{binarize_threshold:.3f}_{time_stamp}.csv'

    model.load_state_dict(torch.load(model_file_name, map_location=device))
    model = model.to(device)
    model.eval()

    dice_coeff_arr = np.zeros(test_dataset.__len__())

    with torch.no_grad():

        for batch_idx, (org_data, org_labels) in enumerate(test_loader):

            data = org_data.to(device)
            outputs = model(data)

            out_mask, label_img = labeling(outputs,org_labels,binarize_threshold)

            dice_coeff_arr[batch_idx] = dice.dice_numpy(out_mask, label_img)

            if export_mask:
                mask_file_name = f'{output_path}/test_images/mask_th{binarize_threshold:.3f}_{batch_idx:05d}.png'
                output = overlay(out_mask,label_img,org_data)
                imageio.imwrite(mask_file_name, output)

            print(f"{batch_idx},{dice_coeff_arr[batch_idx]:.4f}")
            with open(result_file_name, "a") as fp:
                fp.write(f"{batch_idx},{dice_coeff_arr[batch_idx]:.4f}\n")
    eval_vals = dice_coeff_arr
    with open(result_file_name,'a') as fp:
        fp.write("Mean of Dice coefficient: %.4f (%.4f - %.4f)" %
          (np.mean(eval_vals), np.min(eval_vals), np.max(eval_vals)))

    return dice_coeff_arr


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Sample code to test segmentation model in SS2022',
        add_help=True)
    parser.add_argument('test_data_path', help='Path of test data')
    parser.add_argument('model_file_name', help='File name of trained model')
    parser.add_argument('output_path', help='Path of output data')
    parser.add_argument('-g', '--gpu_id', help='GPU IDs',
                        type=str, default='0')
    parser.add_argument('-f', '--first_filter_num',
                        help='Number of the first filter in U-Net',
                        type=int, default=16)
    parser.add_argument('-t', '--binarize_threshold',
                        help='Threshold to binarize outputs',
                        type=float, default=0.5)
    parser.add_argument('--export_mask',
                        help='Export output mask as png file',
                        action='store_true')
    parser.add_argument('--time_stamp', help='Time stamp for saved data',
                        type=str, default='')        
    parser.add_argument('--model', help='Time stamp for saved data',
                        type=int, default=0)                   

    args = parser.parse_args()

    hyperparameters_dict = {"first_filter_num": args.first_filter_num}

    eval_vals = test(args.test_data_path,
                     args.model_file_name,
                     args.output_path,
                     args.binarize_threshold,
                     args.gpu_id,
                     args.export_mask,
                     args.time_stamp,
                     args.model,
                     **hyperparameters_dict)

    print("Mean of Dice coefficient: %.4f (%.4f - %.4f)" %
          (np.mean(eval_vals), np.min(eval_vals), np.max(eval_vals)))

# python sample/test_segmentation.py validation/ output/best_model_best_20221031194537.pth output/ --export_mask