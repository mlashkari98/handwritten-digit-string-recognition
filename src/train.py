import os

import cv2
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss

from dataset import collate_fn, DataGenerator
from model import CRNN
from evaluate import evaluate
from config import train_config as config

import os
from torchvision import io, transforms
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import cv2

PATH = "../ORAND-CAR-2014/ORAND-CAR-2014/CAR-AB/"
SUB_FOLDER = "ab_train_images/"


def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        crnn.parameters(), 5)  # gradient clipping with 5
    optimizer.step()
    return loss.item()


def main():
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    eval_batch_size = config['eval_batch_size']
    lr = config['lr']
    show_interval = config['show_interval']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']

    img_width = config['img_width']
    img_height = config['img_height']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    image_size = (32, 100)

    augmentation = transforms.Compose([
        transforms.Resize(size=(image_size[0], image_size[1])),
    ])

    files, labels = np.loadtxt(
        PATH + "ab_train_labels.txt", delimiter='\t', unpack=True, dtype=str)

    train_files, test_files, train_labels, test_labels = train_test_split(
        files, labels, test_size=0.1, random_state=49)

    train_dataset = DataGenerator(path=PATH, sub_folder=SUB_FOLDER,
                                  images_list=train_files, labels=train_labels, transform=augmentation)

    test_dataset = DataGenerator(path=PATH, sub_folder=SUB_FOLDER,
                                  images_list=test_files, labels=test_labels, transform=augmentation)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=collate_fn)
    valid_loader = DataLoader(
        dataset=test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        # num_workers=cpu_workers,
        collate_fn=collate_fn)

    num_class = len(DataGenerator.LABEL2CHAR) + 1
    crnn_num_classes = 37 #
    crnn = CRNN(1, img_height, img_width, 37,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(
            reload_checkpoint, map_location=device))

        for parameter in crnn.parameters():
            parameter.requires_grad = False

    crnn.dense = torch.nn.Linear(2*256, num_class)
    crnn.to(device)

    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum', zero_infinity=True)
    criterion.to(device)

    assert save_interval % valid_interval == 0
    i = 1
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print('train_batch_loss[', i, ']: ', loss / train_size)
            i += 1
            
        print('train_loss: ', tot_train_loss / tot_train_count)

        if epoch % valid_interval == 0:
            evaluation = evaluate(crnn, valid_loader, criterion,
                                  decode_method=config['decode_method'],
                                  beam_size=config['beam_size'])
            print('valid_evaluation: loss={loss}, acc={acc}'.format(
                **evaluation))

        if epoch % save_interval == 0:
            prefix = 'crnn'
            loss = evaluation['loss']
            save_model_path = os.path.join(config['checkpoints_dir'],
                                           f'{prefix}_{i:06}_loss{loss}.pt')
            torch.save(crnn.state_dict(), save_model_path)
            print('save model at ', save_model_path)


if __name__ == '__main__':
    main()
