"""
Created on Sun Oct 23 2022
@author: ynomura
"""

import argparse
from audioop import avg
from datetime import datetime as dt
import numpy as np
import random as rn
import os
import sys
import torch
import time
import torch.optim as optim
import torch.utils.data
import random
import scipy.ndimage as ndimg
from test_segmentation import *

from dataset_segmentation import DataSetSegmentation
from dataset_segmentation_inside import DataAugmentation
from unet import Unet
from attUnet import attUnet
import dice


def training(training_data_path, validation_data_path, output_path,
             first_filter_num=64, learning_rate=0.001, beta_1=0.99,
             batch_size=16, max_epoch_num=50,binarize_threshold=0.5,
             gpu_id="0", model=0, time_stamp=""):

    # Fix seed

    seed_num = 234567
    os.environ['PYTHONHASHSEED'] = '0'
    rn.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True

    device = f'cuda:{gpu_id}'

    if not os.path.isdir(training_data_path):
        print(f"Error: {training_data_path} is not found.")
        sys.exit(1)

    if not os.path.isdir(validation_data_path):
        print(f"Error: {validation_data_path} is not found.")
        sys.exit(1)

    # Automatic creation of output folder
    if not os.path.isdir(output_path):
        print(f"Path of output data ({output_path}) is created automatically.")
        os.makedirs(output_path)   

    # Set ID of CUDA device
    device = f'cuda:{gpu_id}'
    print(f"Device: {device} First_filter={first_filter_num} Beta={beta_1} LR={learning_rate} model={model} batch={batch_size} epoch_num={max_epoch_num}")

    if time_stamp == "":
            time_stamp = dt.now().strftime('%Y%m%d%H%M%S')
    loss_log_file_name = f"{output_path}/loss_log_{time_stamp}_model:{model}.csv"
    model_file_name = f"{output_path}/model_best_{time_stamp}_model:{model}.pth"

    # DataAugmentation: augment inside, DataSetSegmentation: read pre-created images
    training_dataset = DataAugmentation(training_data_path)
    print('len train:',len(training_dataset))
    training_loader = torch.utils.data.DataLoader(training_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True)

    validation_dataset = DataSetSegmentation(validation_data_path)
    print('len validation:',len(validation_dataset))
    validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                    batch_size=1,
                                                    shuffle=False)
    print("data load finished.")
    # load network
    in_channels = validation_dataset[0][0].shape[0]
    out_channels = validation_dataset[0][1].shape[0]
    print(in_channels, out_channels)

    if model == 0:
        model = Unet(in_channels, out_channels, first_filter_num)
    elif model == 1:
        model = attUnet(in_channels, out_channels, first_filter_num)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),
                            lr=learning_rate,
                            betas=(beta_1, 0.999))
    # optimizer = optim.SGD(model.parameters(),
    #                        lr=learning_rate)
    criterion_D = dice.DiceLoss()
    # criterion_CE = torch.nn.CrossEntropyLoss()

    # training
    best_validation_loss = float('inf')
    dice_coeff_arr = np.zeros(validation_dataset.__len__())

    with open(loss_log_file_name,'a') as f:
        f.write(f"First_filter={first_filter_num} Beta={beta_1} LP={learning_rate} batch={batch_size} epoch_num={max_epoch_num}\n")
    
    for epoch in range(max_epoch_num):
        training_loss = 0
        validation_loss = 0

        # training
        model.train()

        for batch_idx, (data, labels) in enumerate(training_loader):

            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion_D(outputs, labels)

            training_loss += loss.item()

            loss = torch.sum(loss)

            loss.backward()
            optimizer.step()

        avg_training_loss = training_loss / (batch_idx + 1)

        # validation
        model.eval()
        dice_coeff_arr = np.zeros(validation_dataset.__len__())
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(validation_loader):

                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                out_mask, label_img = labeling(outputs,labels,binarize_threshold)

                dice_coeff_arr[batch_idx] = dice.dice_numpy(out_mask, label_img)

                loss = criterion_D(outputs, labels)
                validation_loss += loss.item()

        avg_validation_loss = validation_loss / (batch_idx + 1)

        saved_str = ""

        if best_validation_loss > avg_validation_loss:
            best_validation_loss = avg_validation_loss
            torch.save(model.state_dict(), model_file_name)
            saved_str = " ==> model saved"

        eval_vals = dice_coeff_arr
        print("epoch %d: train_loss:%.4f val_loss:%.4f %s dice:%.4f (%.4f - %.4f)" %
            (epoch + 1, avg_training_loss, avg_validation_loss, saved_str, 
            np.mean(eval_vals), np.min(eval_vals), np.max(eval_vals)))
        with open(loss_log_file_name, "a") as fp:
            fp.write("%d,%.4f,%.4f,%.4f\n" %
                    (epoch + 1, avg_training_loss, avg_validation_loss, np.mean(eval_vals)))

    return model_file_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample code for training segmentation model in SS2022',
                                     add_help=True)
    parser.add_argument('training_data_path', help='Path of training data')
    parser.add_argument('validation_data_path', help='Path of validation data')
    parser.add_argument('output_path',
                        help='Path of output data')
    parser.add_argument('-g', '--gpu_id', help='GPU ID',
                        type=str, default='0')
    parser.add_argument('-f', '--first_filter_num',
                        help='Number of the first filter in U-Net',
                        type=int, default=16)
    parser.add_argument('-l', '--learning_rate',
                        help='Learning rate',
                        type=float, default=0.01)
    parser.add_argument('--beta_1',
                        help='Beta_1 for Adam',
                        type=float, default=0.9)
    parser.add_argument('-b', '--batch_size',
                        help='Batch size',
                        type=int, default=8)
    parser.add_argument('-m', '--max_epoch_num',
                        help='Maximum number of training epochs',
                        type=int, default=50)
    parser.add_argument('-t', '--binarize_threshold',
                        help='Threshold to binarize outputs',
                        type=float, default=0.5)
    parser.add_argument('-mo', '--model',
                        help='Threshold to binarize outputs',
                        type=int, default=1)
    parser.add_argument('--time_stamp', help='Time stamp for saved data',
                        type=str, default='')

    args = parser.parse_args()

    for round in range(1):
        random.seed(time.time())
        r_lr = 10**random.randint(-6,-2)
        r_f = 2**random.randint(2,4)
        r_beta = random.uniform(0.9,0.99)
        r_batch = 2**random.randint(1,3)
        print('Round:',round+1)
        training(args.training_data_path,
             args.validation_data_path,
             args.output_path,
             r_f,
             r_lr,
             r_beta,
             r_batch,
             args.max_epoch_num,
             args.binarize_threshold,
             args.gpu_id,
             args.model,
             args.time_stamp)

# python training_segmentation.py training/ validation/ output/
# best para without Att: First_filter=16 Beta=0.9693796803178418 LP=0.0001 batch=2