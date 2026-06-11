#!pip install torch torchvision thop tqdm

#----------------------Merged-----------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
import argparse
from torch.nn.parallel import DataParallel
from thop import profile, clever_format
from torch.utils.data import Subset
from torchvision import datasets, transforms
import threading
import queue
import warnings
import csv
import math
warnings.filterwarnings("ignore")


# ============================== CWConv ==============================
class CWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, num_class=10, dropout=0):
        super(CWConv, self).__init__()
        assert out_channels % num_class == 0
        self.num_class = num_class
        self.out_channels = out_channels
        self.current_weights = None

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        torch.nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def update_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, x, no_norm=False):
        x = x.detach()
        y = self.conv(x)
        y = self.relu(y)
        y = self.dropout(y)

        g = y.view(y.size(0), self.num_class, -1)
        g = g.mean(dim=2)
        if no_norm:
            return y, g

        y = F.group_norm(y, self.num_class)
        return y, g


# ============================== Dataset ==============================
def get_cifar100_dataloader(root, train_batch_size=128, test_batch_size=128, seed=2222):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=root, train=False, transform=transform, download=True)

    torch.manual_seed(seed)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [45000, 5000])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=train_batch_size, shuffle=False, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=8)

    return train_loader, valid_loader, test_loader


# ============================== ResNet Model ==============================
NO_SHORTCUT = 0
ADD_SHORTCUT = 1
CONCAT_SHORTCUT = 2

loss_layers = []

class Resnet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_class=100,
                 planes=(100, 200, 400, 800),
                 dropout=0.1,
                 bias=False,
                 learning_rate=0.08,
                 lr_min=0.004,
                 weight_decay=0.,
                 devices=None,
                 epochs=150
                 ):
        super(Resnet, self).__init__()
        self.num_class = num_class

        self.input_shortcut_flag = [True]
        self.shortcut_flag = [NO_SHORTCUT]
        self.downsample_flag = [False]
        for i in range(4):
            self.input_shortcut_flag.extend([False, True, False, True])
            self.shortcut_flag.extend([NO_SHORTCUT, ADD_SHORTCUT, NO_SHORTCUT, CONCAT_SHORTCUT])
            self.downsample_flag.extend([False, False, False, True])

        self.input_shortcut_flag[-1] = False

        self.layers = nn.ModuleList([
            CWConv(in_channels=in_channels, out_channels=planes[0], kernel_size=3, stride=1, padding=1, bias=bias,  #0
                   num_class=num_class, dropout=dropout),
            CWConv(planes[0], planes[0], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),  #1
            CWConv(planes[0], planes[0], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),  #2
            CWConv(planes[0], planes[0], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),  #3
            CWConv(planes[0], planes[0], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),  #4
            CWConv(planes[1], planes[1], 3, stride=2, padding=1, bias=bias, num_class=num_class, dropout=dropout), #5
            CWConv(planes[1], planes[1], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout), #6
            CWConv(planes[1], planes[1], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout), #7
            CWConv(planes[1], planes[1], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout), #8
            CWConv(planes[2], planes[2], 3, stride=2, padding=1, bias=bias, num_class=num_class, dropout=dropout), #9
            CWConv(planes[2], planes[2], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),  #10
            CWConv(planes[2], planes[2], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),  #11
            CWConv(planes[2], planes[2], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),  #12
            CWConv(planes[3], planes[3], 3, stride=2, padding=1, bias=bias, num_class=num_class, dropout=dropout),  #13
            CWConv(planes[3], planes[3], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),  #14
            CWConv(planes[3], planes[3], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),  #15
            CWConv(planes[3], planes[3], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),  #16
        ])

        if devices is not None:
            for i, dev in enumerate(devices):
                for j in range(i, len(self.layers), len(devices)):
                    if j < len(self.layers):
                        self.layers[j].to(dev)

        self.optimizers = [torch.optim.AdamW(layer.parameters(), lr=learning_rate, weight_decay=weight_decay)
                           for layer in self.layers]
        self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_min)
                           for opt in self.optimizers]

        self.register_buffer('start_layer', torch.tensor(1, dtype=torch.int))
        self.register_buffer('end_layer', torch.tensor(16, dtype=torch.int))

    def forward(self, x, layer_idx=-1, no_norm=False):
        devices = [next(p.device for p in layer.parameters()) for layer in self.layers]
        g_cls = None
        x = F.layer_norm(x, x.shape[1:])
        shortcut = torch.zeros(1)

        for i, layer in enumerate(self.layers):
            if i > self.end_layer:
                break
            x = x.to(devices[i])

            if i == layer_idx:
                x, _ = layer(x, no_norm=no_norm)
                return x

            x, g = layer(x)
            if self.start_layer < i <= self.end_layer:
                g_cls = g_cls + g.to(devices[0]) if g_cls is not None else g.to(devices[0])
            elif i == self.start_layer:
                g_cls = g.to(devices[0])

            if self.shortcut_flag[i] == ADD_SHORTCUT:
                x += shortcut.to(x.device)
            elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                x = torch.cat([x, shortcut.to(x.device)], dim=1)
            if self.input_shortcut_flag[i]:
                shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()

        return g_cls

    def update(self, dataloader, task_dir, epoch):
        global loss_layers
        loss_layers = []
        devices = [next(p.device for p in layer.parameters()) for layer in self.layers]
        criterion = nn.CrossEntropyLoss()
        shortcut = torch.zeros(1)

        self.train()
        for x, labels in dataloader:
            x, labels = x.to(devices[0]), labels.to(devices[0])
            x = F.layer_norm(x, x.shape[1:])

            for i, layer in enumerate(self.layers):
                x, labels = x.to(devices[i]), labels.to(devices[i])
                self.optimizers[i].zero_grad()
                x, g = layer(x)
                loss = criterion(g, labels)
                loss_layers.append(loss.item())
                loss.backward()
                self.optimizers[i].step()

                if self.shortcut_flag[i] == ADD_SHORTCUT:
                    x += shortcut.to(x.device)
                elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                    x = torch.cat([x, shortcut.to(x.device)], dim=1)

                if self.input_shortcut_flag[i]:
                    shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()

        for sch in self.schedulers:
            sch.step()

    def pruning(self, dataloader):
        devices = [next(p.device for p in layer.parameters()) for layer in self.layers]
        corrects = [[0 for _ in range(len(self.layers))] for _ in range(len(self.layers))]
        total = 0
        self.eval()

        with torch.no_grad():
            for x, labels in dataloader:
                x, labels = x.to(devices[0]), labels.to(devices[0])
                total += labels.size(0)
                x = F.layer_norm(x, x.shape[1:])
                shortcut = torch.zeros(1)
                gs_cls = [None] * len(self.layers)

                for i, layer in enumerate(self.layers):
                    x = x.to(devices[i])
                    x, g = layer(x)
                    for j in range(i + 1):
                        gs_cls[j] = (gs_cls[j] + g.to(devices[0]) if gs_cls[j] is not None else g.to(devices[0]))
                        pred = torch.argmax(gs_cls[j], dim=1)
                        corrects[j][i] += torch.eq(pred, labels).sum().item()

                    if self.shortcut_flag[i] == ADD_SHORTCUT:
                        x += shortcut.to(x.device)
                    elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                        x = torch.cat([x, shortcut.to(x.device)], dim=1)
                    if self.input_shortcut_flag[i]:
                        shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()

        best_pred = 0
        best_start = 0
        best_end = 0
        for j in range(len(self.layers)):
            for i in range(j + 1):
                if corrects[i][j] > best_pred:
                    best_pred = corrects[i][j]
                    best_start = i
                    best_end = j

        self.start_layer = torch.tensor(best_start, dtype=torch.int)
        self.end_layer = torch.tensor(best_end, dtype=torch.int)

        full_acc = corrects[1][-1] / total * 100
        best_acc = best_pred / total * 100
        return best_acc, full_acc

    def test_local_acc(self, dataloader):
        devices = [next(p.device for p in layer.parameters()) for layer in self.layers]
        corrects = [0] * len(self.layers)
        total = 0
        self.eval()
        with torch.no_grad():
            for x, labels in dataloader:
                x, labels = x.to(devices[0]), labels.to(devices[0])
                total += labels.size(0)
                x = F.layer_norm(x, x.shape[1:])
                shortcut = torch.zeros(1)
                for i, layer in enumerate(self.layers):
                    x = x.to(devices[i])
                    _, g = layer(x)
                    pred = torch.argmax(g, dim=1)
                    corrects[i] += torch.eq(pred, labels).sum().item()

                    if self.shortcut_flag[i] == ADD_SHORTCUT:
                        x += shortcut.to(x.device)
                    elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                        x = torch.cat([x, shortcut.to(x.device)], dim=1)
                    if self.input_shortcut_flag[i]:
                        shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()

        return [100 * c / total for c in corrects]


def test_model(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, labels in test_loader:
            x, labels = x.to(device), labels.to(device)
            y = model(x)
            pred = torch.argmax(y, dim=1)
            correct += torch.eq(pred, labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


# ============================== Main Training Loop ==============================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DeepForward Training')
    args = argparse.Namespace(
        task='cifar10',
        epochs=150,
        train_batch_size=128,
        test_batch_size=128,
        lr=0.08,
        lr_min=0.008,
        weight_decay=5e-4,
        dropout=0.2,
        devices=[0],
        seed=2222,
        task_dir=None,
        arch='resnet',
        parallel=False,
        save_step=5,
        queue_size=5,
        data_dir='./data'
    )

    # === Config ===
    epochs = args.epochs
    task_dir = args.task_dir or os.path.join('./results/', f'{time.strftime("%Y%m%d-%H%M%S")}_kham_{args.task}')
    os.makedirs(task_dir, exist_ok=True)

    devices = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else 'cpu') for i in args.devices]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, valid_loader, test_loader = get_cifar100_dataloader(root=args.data_dir,
                                                                      train_batch_size=args.train_batch_size,
                                                                      test_batch_size=args.test_batch_size,
                                                                      seed=args.seed)

    num_class = 100
    model = Resnet(in_channels=3, num_class=num_class, planes=[100,200,400,800],
                   dropout=args.dropout, learning_rate=args.lr, lr_min=args.lr_min,
                   weight_decay=args.weight_decay, devices=devices, epochs=epochs)

    # === Info CSV Setup (like Analysis script) ===
    info_csv_path = os.path.join(task_dir, "info_17L.csv")
    with open(info_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "start_layer", "end_layer",
            "before_acc", "valid_acc", "test_acc",
            "train_time", "pruning_time", "test_time", "lr", "info_str"
        ])

    # FLOPs
    input_data = torch.randn(1, 3, 32, 32, device=devices[0])
    flops, params = profile(model, inputs=(input_data,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f'FLOPs: {flops}, Params: {params}')

    for epoch in range(args.epochs):
        for layer in model.layers:
            layer.update_epoch(epoch)

        start_time = time.time()
        model.update(train_loader, task_dir, epoch)
        train_time = time.time() - start_time

        pruning_time = time.time()
        valid_acc, before_acc = model.pruning(valid_loader)
        pruning_time = time.time() - pruning_time

        test_time = time.time()
        test_acc = test_model(model, test_loader, devices[0])
        test_time = time.time() - test_time

        current_lr = model.optimizers[0].param_groups[0]["lr"]

        info = (f'Epoch: {(epoch + 1):03d}/{epochs:03d}: '
                f'Pruning ({model.start_layer:02d}->{model.end_layer:02d}):\t'
                f'Valid Acc:{before_acc:.2f}% -> {valid_acc:.2f}%\t '
                f'Test Acc: {test_acc:.2f}% \t || '
                f'Train Time: {train_time:.2f}s, '
                f'Pruning Time: {pruning_time:.2f}s '
                f'Test Time: {test_time:.2f}s '
                f'|| lr: {current_lr:.5f}')

        print(info)

        # === Save to info CSV ===
        with open(info_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                model.start_layer.item(),
                model.end_layer.item(),
                round(before_acc, 2),
                round(valid_acc, 2),
                round(test_acc, 2),
                round(train_time, 2),
                round(pruning_time, 2),
                round(test_time, 2),
                round(current_lr, 5),
                info.replace(",", ";")  # avoid CSV conflicts
            ])

        # Optional: save accuracy log
        if epoch % args.save_step == 0 or epoch == epochs - 1:
            with open(os.path.join(task_dir, 'accuracy.csv'), 'a') as f:
                f.write(f'{epoch+1},{before_acc:.2f},{valid_acc:.2f},{test_acc:.2f}\n')

        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': [opt.state_dict() for opt in model.optimizers],
                'scheduler_state_dict': [sch.state_dict() for sch in model.schedulers],
            }, os.path.join(task_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    print("Training completed. Info logged to:", info_csv_path)