import time
import os
import torch
import math
from torch.nn.parallel import DataParallel
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms
import csv
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings("ignore")

class CWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=False, num_class=10, dropout=0, reg_ortho=1e-3,
                 reg_entropy=1e-2, temperature=10, temp_decay_rate=0.995):
        super().__init__()

        assert out_channels % num_class == 0
        self.num_class = num_class
        self.out_channels = out_channels

        self.current_weights = None  # To store the latest weights
        self.current_ortho_loss = None  # To store the latest orthogonality loss

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.class_channel_weights = nn.Parameter(
            torch.ones(out_channels, num_class)*0.5
        )

        self.reg_entropy = reg_entropy
        self.reg_ortho = reg_ortho
        self.temperature = temperature
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        self.temp_decay_rate = temp_decay_rate
        self.epoch = 0

        self.channels_per_class = self.out_channels // self.num_class
        self.remainder = self.out_channels % self.num_class

    def update_epoch(self, epoch):
        """Update epoch for temperature scheduling"""
        self.epoch = epoch

    def forward(self, x, no_norm=False, return_reg=False):
        x = x.detach()
        y = self.conv(x)
        y = self.relu(y)
        y = self.dropout(y)

        B, C, H, W = y.shape
        y_pooled = 0.97 * y.mean(dim=[2,3]) + 0.03 * y.amax(dim=[2,3]) - 0.03 * y.amin(dim=[2,3])
        y_squared = y_pooled ** 1

        base_temp = torch.exp(self.log_temperature).clamp(min=0.1, max=100)
        temperature = base_temp * (self.temp_decay_rate ** self.epoch)
        weights = F.softmax(self.class_channel_weights / temperature, dim=1)

        g = y_squared @ weights

        if no_norm:
            return y, g

        y = F.instance_norm(y)
        self.current_weights = weights.detach().cpu()

        if return_reg:
            ent = -torch.sum(weights * torch.log(weights + 1e-10), dim=0).mean()
            loss_ent = self.reg_entropy * ent
            
            # Normalize weights
            weight_norm = weights / (weights.norm(dim=0, keepdim=True) + 1e-10)
            gram = weight_norm.T @ weight_norm
            identity = torch.eye(self.num_class, device=weights.device)
            # Normalize by matrix size and channel count
            ortho_loss = (1.0 / (self.num_class ** 2 * self.out_channels)) * ((gram - identity) ** 2).mean()
            # Curriculum-based scaling (warmup over 50 epochs)
            curriculum_factor = min(1.0, self.epoch / 50.0)
            loss_ortho = self.reg_ortho * ortho_loss * curriculum_factor
            self.current_ortho_loss = ortho_loss.detach().cpu()
            
            return y, g, loss_ent + loss_ortho

        return y, g

    def get_orthogonality_score(self):
        """Compute orthogonality score as Frobenius norm of Gram matrix deviation from identity"""
        weights = F.softmax(self.class_channel_weights / torch.exp(self.log_temperature).clamp(min=0.1, max=100), dim=1)
        weight_norm = weights / (weights.norm(dim=0, keepdim=True) + 1e-10)
        gram = weight_norm.T @ weight_norm
        identity = torch.eye(self.num_class, device=gram.device)
        ortho_score = torch.norm(gram - identity, p='fro').item()
        return ortho_score

#---------------------------------Dataset--------------------------------#

filename_postfix = "17L"
info_csv_path = f"info_{filename_postfix}.csv"
ortho_csv_path = f"ortho_metrics_{filename_postfix}.csv"

# Create the info CSV
with open(info_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch",
        "start_layer",
        "end_layer",
        "before_acc",
        "valid_acc",
        "before_test_acc",
        "test_acc",
        "train_time",
        "pruning_time",
        "test_time",
        "lr",
        "info_str"
    ])

#----------------------------------MNIST-----------------------------------#

def get_mnist_dataloader(root, train_batch_size=128, test_batch_size=128, seed=2222, valid=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),  # 归一化
        transforms.Resize((32, 32))
    ])

    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform, download=True)
    valid_dataset = None

    if valid:
        torch.manual_seed(seed)
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8
    )

    valid_loader = None
    if valid:
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=8
        )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=8
    )

    if valid:
        return train_loader, valid_loader, test_loader
    return train_loader, test_loader

#----------------------------------Resnet-----------------------------------#

import threading
import queue

NO_SHORTCUT = 0
ADD_SHORTCUT = 1
CONCAT_SHORTCUT = 2

loss_layers = []
loss_layers_dict = {f'{i}':[] for i in range(17)}
current_layer = 0
current_epoch = 0

class Resnet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_class=10,
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
            CWConv(planes[0], planes[0], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3),  #1
            CWConv(planes[0], planes[0], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3),  #2
            CWConv(planes[0], planes[0], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3),  #3
            CWConv(planes[0], planes[0], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3),  #4
            CWConv(planes[1], planes[1], 3, stride=2, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3), #5
            CWConv(planes[1], planes[1], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3), #6
            CWConv(planes[1], planes[1], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3), #7
            CWConv(planes[1], planes[1], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3), #8
            CWConv(planes[2], planes[2], 3, stride=2, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3), #9
            CWConv(planes[2], planes[2], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3),  #10
            CWConv(planes[2], planes[2], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3),  #11
            CWConv(planes[2], planes[2], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3),  #12
            CWConv(planes[3], planes[3], 3, stride=2, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3),  #13
            CWConv(planes[3], planes[3], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3),  #14
            CWConv(planes[3], planes[3], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3),  #15
            CWConv(planes[3], planes[3], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout, reg_ortho=1e-3),  #16
        ])

        if devices is not None:
            num_layer = len(self.layers)
            num_device = len(devices)
            layer_groups = [i * num_layer // num_device for i in range(num_device)]
            layer_groups.append(num_layer)

            for i in range(num_device):
                for j in range(layer_groups[i], layer_groups[i + 1]):
                    self.layers[j].to(devices[i])

        lr_boost = 1.01
        self.optimizers = []
        for i, layer in enumerate(self.layers):
            self.optimizers.append(torch.optim.AdamW(layer.parameters(), lr=learning_rate, weight_decay=weight_decay))
            
        self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
                           for optimizer in self.optimizers]

        self.register_buffer('start_layer', torch.tensor(1, dtype=torch.int))
        self.register_buffer('end_layer', torch.tensor(16, dtype=torch.int))

    def get_weight_distribution(self, layer_idx, bins=50):
        """Get histogram of weight distribution for a specific layer's convolutional weights."""
        weights = self.layers[layer_idx].conv.weight.data.cpu().numpy().flatten()
        hist, bin_edges = np.histogram(weights, bins=bins, density=True)
        return hist, bin_edges

    def save_weight_distribution(self, epoch, task_dir, bins=50):
        """Append weight distribution histograms for all layers to a single CSV file."""
        dist_data = []
        for i in range(len(self.layers)):
            hist, bin_edges = self.get_weight_distribution(i, bins)
            for j, (h, be) in enumerate(zip(hist, bin_edges[:-1])):
                dist_data.append({
                    'epoch': epoch,
                    'layer': i,
                    'bin_edge': be,
                    'histogram': h
                })
        df = pd.DataFrame(dist_data)
        hist_path = os.path.join(task_dir, 'weight_dist.csv')
        mode = 'a' if os.path.exists(hist_path) else 'w'
        df.to_csv(hist_path, mode=mode, index=False, header=not os.path.exists(hist_path))

    def forward(self, x, layer_idx=-1, no_norm=False):
        devices = [next(layer.parameters()).device for layer in self.layers]

        g_cls = None
        x = F.layer_norm(x, x.shape[1:])

        shortcut = torch.zeros(1)

        for i, layer in enumerate(self.layers):
            if i > self.end_layer:
                break
            x = x.to(devices[i])

            # return local representation
            if i == layer_idx:
                x, _ = layer(x, no_norm=no_norm)
                return x

            x, g = layer(x)

            if self.start_layer < i <= self.end_layer:
                g_cls += g.to(devices[0])
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
        global loss_layer
        global loss_layers_dict
        global current_layer
        global losses_filename
        global current_epoch

        loss_layers_dict = {f'{i}':[] for i in range(17)}
        loss_layers = []
        devices = [next(layer.parameters()).device for layer in self.layers]

        shortcut = torch.zeros(1)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.train()
        current_iter_loss = []
        for x, labels in dataloader:
            x, labels = x.to(devices[0]), labels.to(devices[0])
            x = F.layer_norm(x, x.shape[1:])
            for i, layers in enumerate(self.layers):
                if i > self.end_layer:
                    break
                
                x, labels = x.to(devices[i]), labels.to(devices[i])
                self.optimizers[i].zero_grad()
                x, g, reg_loss = layers(x, return_reg=True)  # Get regularization loss
                loss = criterion(g, labels) + reg_loss  # Add regularization
                loss_layers_dict[str(i)].append(criterion(g, labels).item())  # Store the loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(layers.parameters(), 1.0)
                self.optimizers[i].step()

                if self.shortcut_flag[i] == ADD_SHORTCUT:
                    x += shortcut.to(x.device)
                elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                    x = torch.cat([x, shortcut.to(x.device)], dim=1)

                if self.input_shortcut_flag[i]:
                    shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()

        for scheduler in self.schedulers:
            scheduler.step()

        temp_dict = loss_layers_dict.copy()
        if len(loss_layers_dict['0']) > 0:
            for i in range(len(loss_layers_dict)):
                layer_loss_mean = (torch.tensor(loss_layers_dict[str(i)], dtype=float).mean(dim=0)).item()
                loss_layers_dict[str(i)] = []
                temp_dict[str(i)] = layer_loss_mean

            with open(losses_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                if current_epoch > 0:
                    writer.writerow(temp_dict.values())

        loss_layers_dict = dict(temp_dict)

        # Save weight distribution after each epoch
        self.save_weight_distribution(epoch, task_dir)

    def pruning(self, dataloader, test_dataloader, k=5):
        devices = [next(layer.parameters()).device for layer in self.layers]
        self.eval()
        total_val = 0
        total_test = 0
        full_val_correct = 0
        full_test_correct = 0  # New variable for test accuracy

        # ----------- Part 1: Accuracy with all layers (for before_acc) -----------
        with torch.no_grad():
            for x, labels in dataloader:
                shortcut = torch.zeros(1)
                x, labels = x.to(devices[0]), labels.to(devices[0])
                total_val += labels.size(0)
                x = F.layer_norm(x, x.shape[1:])

                g_cls = None
                for i, layer in enumerate(self.layers):
                    x = x.to(devices[i])
                    x, g = layer(x)

                    if self.start_layer < i <= self.end_layer:
                        g_cls += g.to(devices[0])
                    elif i == self.start_layer:
                        g_cls = g.to(devices[0])

                    # shortcut logic
                    if self.shortcut_flag[i] == ADD_SHORTCUT:
                        x += shortcut.to(x.device)
                    elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                        x = torch.cat([x, shortcut.to(x.device)], dim=1)
                    if self.input_shortcut_flag[i]:
                        shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()

                pred = torch.argmax(g_cls, dim=1)
                full_val_correct += torch.eq(pred, labels).sum().item()

            # Test accuracy (before_test_acc)
            for x, labels in test_dataloader:
                shortcut = torch.zeros(1)
                x, labels = x.to(devices[0]), labels.to(devices[0])
                total_test += labels.size(0)
                x = F.layer_norm(x, x.shape[1:])

                g_cls = None
                for i, layer in enumerate(self.layers):
                    x = x.to(devices[i])
                    x, g = layer(x)

                    if self.start_layer < i <= self.end_layer:
                        g_cls += g.to(devices[0])
                    elif i == self.start_layer:
                        g_cls = g.to(devices[0])

                    # shortcut logic
                    if self.shortcut_flag[i] == ADD_SHORTCUT:
                        x += shortcut.to(x.device)
                    elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                        x = torch.cat([x, shortcut.to(x.device)], dim=1)
                    if self.input_shortcut_flag[i]:
                        shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()

                pred = torch.argmax(g_cls, dim=1)
                full_test_correct += torch.eq(pred, labels).sum().item()
    
        before_val_acc = 100.0 * full_val_correct / total_val
        before_test_acc = 100.0 * full_test_correct / total_test  # Compute before_test_acc

        devices = [next(layer.parameters()).device for layer in self.layers]
        self.eval()

        correct = 0

        # ---------- Step 1: Compute average losses ----------
        avg_losses = []
        for i in range(len(self.layers)):
            if loss_layers_dict[str(i)]:
                avg_loss = torch.tensor(loss_layers_dict[str(i)], dtype=torch.float).mean().item()
            else:
                avg_loss = float('inf')
            avg_losses.append(avg_loss)

        # ---------- Step 2: Compute softmax weights ----------
        losses_tensor = torch.tensor(avg_losses, dtype=torch.float)
        weights = torch.softmax(-losses_tensor, dim=0)  # negative because lower loss is better

        # ---------- Step 3: Run prediction with weighted g_cls ----------
        with torch.no_grad():
            for x, labels in dataloader:
                x, labels = x.to(devices[0]), labels.to(devices[0])
                x = F.layer_norm(x, x.shape[1:])

                shortcut = torch.zeros(1)
                g_cls = torch.zeros(x.size(0), self.num_class).to(devices[0])

                for i, layer in enumerate(self.layers):
                    x = x.to(devices[i])
                    x, g = layer(x)

                    weight = weights[i].to(devices[0])
                    g_cls += weight * g.to(devices[0])  # weighted accumulation

                    # shortcut logic
                    if self.shortcut_flag[i] == ADD_SHORTCUT:
                        x += shortcut.to(x.device)
                    elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                        x = torch.cat([x, shortcut.to(x.device)], dim=1)
                    if self.input_shortcut_flag[i]:
                        shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()

                pred = torch.argmax(g_cls, dim=1)
                correct += torch.eq(pred, labels).sum().item()

        correct_test = 0

        with torch.no_grad():
            for x, labels in test_dataloader:
                x, labels = x.to(devices[0]), labels.to(devices[0])
                x = F.layer_norm(x, x.shape[1:])

                shortcut = torch.zeros(1)
                g_cls = torch.zeros(x.size(0), self.num_class).to(devices[0])

                for i, layer in enumerate(self.layers):
                    x = x.to(devices[i])
                    x, g = layer(x)

                    weight = weights[i].to(devices[0])
                    g_cls += weight * g.to(devices[0])  # weighted accumulation

                    # shortcut logic
                    if self.shortcut_flag[i] == ADD_SHORTCUT:
                        x += shortcut.to(x.device)
                    elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                        x = torch.cat([x, shortcut.to(x.device)], dim=1)
                    if self.input_shortcut_flag[i]:
                        shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()

                pred = torch.argmax(g_cls, dim=1)
                correct_test += torch.eq(pred, labels).sum().item()

        valid_acc = 100.0 * correct / total_val
        test_acc = 100.0 * correct_test / total_test
        print(f"[Pruning] Weighted Softmax Layer Accuracy: {valid_acc:.2f}%")

        # Optionally update start/end layer range to cover full model
        self.start_layer = torch.tensor(0, dtype=torch.int)
        self.end_layer = torch.tensor(len(self.layers) - 1, dtype=torch.int)

        return valid_acc, before_val_acc, test_acc, before_test_acc

    def update_pipeline(self, dataloader, queue_size=5, task_dir=None, epoch=None):
        def train_func(module, optimizer, dataloader_size, in_queue, out_queue, in_shortcut_queue, out_shortcut_queue, shortcut_flag):
            device = next(module.parameters()).device
            criterion = nn.CrossEntropyLoss()
            module.train()
            for _ in range(dataloader_size):
                x, labels = in_queue.get()
                x, labels = x.to(device), labels.to(device)

                optimizer.zero_grad()
                x, g = module(x)

                if in_shortcut_queue is not None:
                    shortcut = in_shortcut_queue.get().to(device)
                    if shortcut.shape[2:] != x.shape[2:]:
                        shortcut = F.avg_pool2d(shortcut, 2, stride=2).detach()
                    if shortcut_flag == ADD_SHORTCUT:
                        x += shortcut
                    elif shortcut_flag == CONCAT_SHORTCUT:
                        x = torch.cat([x, shortcut], dim=1)

                if out_queue is not None:
                    out_queue.put((x, labels))
                if out_shortcut_queue is not None:
                    out_shortcut_queue.put(x.detach())

                loss = criterion(g, labels)
                loss.backward()
                optimizer.step()

        self.train()
        main_queues = [queue.Queue(queue_size) for _ in range(len(self.layers))]
        shortcut_queues = [queue.Queue(queue_size) if self.shortcut_flag[i] != NO_SHORTCUT else None
                          for i in range(len(self.layers))]
        main_queues += [None]
        shortcut_queues += [None, None]

        threads = [threading.Thread(target=train_func,
                                   args=(self.layers[i], self.optimizers[i], len(dataloader), main_queues[i],
                                         main_queues[i+1], shortcut_queues[i], shortcut_queues[i+2], self.shortcut_flag[i]))
                   for i in range(len(self.layers))]

        for t in threads:
            t.start()

        for x, labels in dataloader:
            x = x.to(next(self.layers[0].parameters()).device)
            x = F.layer_norm(x, x.shape[1:])
            main_queues[0].put((x, labels))

        for t in threads:
            t.join()

        for scheduler in self.schedulers:
            scheduler.step()

        # Save weight distribution after each epoch
        if task_dir and epoch is not None:
            self.save_weight_distribution(epoch, task_dir)

    def test_local_acc(self, dataloader):
        devices = [next(layer.parameters()).device for layer in self.layers]
        corrects = [0 for _ in range(len(self.layers))]
        self.eval()
        total = 0
        with torch.no_grad():
            for x, labels in dataloader:
                shortcut = torch.zeros(1)
                x, labels = x.to(devices[0]), labels.to(devices[0])
                total += labels.size(0)
                x = F.layer_norm(x, x.shape[1:])

                for i, layer in enumerate(self.layers):
                    x, labels = x.to(devices[i]), labels.to(devices[i])
                    x, g = layer(x)
                    pred = torch.argmax(g, dim=1)
                    corrects[i] += torch.eq(pred, labels).sum().float().item()

                    if self.shortcut_flag[i] == ADD_SHORTCUT:
                        x += shortcut.to(x.device)
                    elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                        x = torch.cat([x, shortcut.to(x.device)], dim=1)
                    if self.input_shortcut_flag[i]:
                        shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()

        return [100 * corrects[i] / total for i in range(len(self.layers))]

    def save_orthogonality_metrics(self, epoch, filename=ortho_csv_path):
        """Save orthogonality loss and score for each layer"""
        ortho_losses = []
        ortho_scores = []
        for layer in self.layers:
            ortho_loss = layer.current_ortho_loss.item() if layer.current_ortho_loss is not None else 0.0
            ortho_score = layer.get_orthogonality_score()
            ortho_losses.append(ortho_loss)
            ortho_scores.append(ortho_score)
        
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch] + ortho_losses + ortho_scores)

    def print_layer_weights(self):
        print("\nLayer Weights (Sparsemax outputs):")
        print("Layer\tClass Weights (sum to 1 per class)")
        print("-" * 60)
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'current_weights') and layer.current_weights is not None:
                weights = layer.current_weights
                print(f"Layer {i}:")
                for class_idx in range(weights.shape[1]):
                    class_weights = weights[:, class_idx]
                    print(f"  Class {class_idx}: {class_weights.numpy().round(4)}")
            else:
                print(f"Layer {i}: No weights available")
        print("-" * 60)

    def save_layer_weights(self, filename="all_epochs_layer_weights.csv", epoch=None, stats_filename="layer_weight_stats.csv"):
        """Save sparsemax weights for all layers to a CSV file, appending each epoch.
        Also compute and save statistics for layer weights and channel weights."""
        # --- determine maximum flattened channel length among all layers (so CSV has fixed columns) ---
        max_ch_len = 0
        for layer in self.layers:
            w = layer.conv.weight.detach().cpu()
            # per out-channel flattened length = in_channels * kernel_h * kernel_w
            ch_len = int(w.shape[1] * w.shape[2] * w.shape[3])
            if ch_len > max_ch_len:
                max_ch_len = ch_len

        # --- prepare header for weights CSV ---
        headers = [
            "epoch",              # training epoch
            "layer_idx",          # index of layer in self.layers
            "out_channel_idx",    # which out_channel within the layer
            "num_out_channels",   # total out_channels in that layer
            "channel_size",       # number of weights in this out_channel (ch_length)
            "channel_mean"        # mean value of the flattened weights for this out_channel
        ]

        # open file in append or write mode
        append_mode = 'a' if os.path.exists(filename) else 'w'
        with open(filename, append_mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            if append_mode == 'w':
                writer.writerow(headers)

            # write rows: one row per layer per out_channel
            for i, layer in enumerate(self.layers):
                conv_w = layer.conv.weight.detach().cpu().numpy()  # shape: (out_c, in_c, kH, kW)
                out_c = conv_w.shape[0]
                ch_len = int(conv_w.shape[1] * conv_w.shape[2] * conv_w.shape[3])

                for out_idx in range(out_c):
                    vec = conv_w[out_idx].flatten()
                    ch_mean = float(np.mean(vec))

                    row = [epoch, i, out_idx, out_c, ch_len, ch_mean]
                    writer.writerow(row)

        # --- write per-layer stats to stats_filename (compact summary) ---
        stats_header = [
            "epoch", "layer_idx",
            "conv_mean", "conv_std", "conv_min", "conv_max",
            "l2_norm", "grad_norm"
        ]
        stats_append_mode = 'a' if os.path.exists(stats_filename) else 'w'
        with open(stats_filename, stats_append_mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            if stats_append_mode == 'w':
                writer.writerow(stats_header)

            for i, layer in enumerate(self.layers):
                conv_w = layer.conv.weight.detach().cpu().numpy().flatten()
                conv_mean = float(np.mean(conv_w))
                conv_std = float(np.std(conv_w))
                conv_min = float(np.min(conv_w))
                conv_max = float(np.max(conv_w))
                l2_norm = float(np.linalg.norm(conv_w))

                # gradient norm (total for layer)
                grad_norm = 0.0
                for p in layer.parameters():
                    if p.grad is not None:
                        grad_norm += float(p.grad.detach().cpu().data.norm(2).item()) ** 2
                grad_norm = math.sqrt(grad_norm)

                writer.writerow([epoch, i, conv_mean, conv_std, conv_min, conv_max, l2_norm, grad_norm])
        
        # Additional useful info: per-layer accuracy on validation set every 10 epochs
        if epoch % 10 == 0 or epoch == 0:
            valid_layer_acc = self.test_local_acc(valid_loader)
            acc_filename = filename.replace("all_epochs_layer_weights.csv", "per_layer_acc.csv")
            acc_header = ["epoch"] + [f"layer_{i}" for i in range(len(valid_layer_acc))]
            
            append_mode = 'a' if os.path.exists(acc_filename) else 'w'
            with open(acc_filename, append_mode, newline='') as csvfile:
                writer = csv.writer(csvfile)
                if append_mode == 'w':
                    writer.writerow(acc_header)
                writer.writerow([epoch] + valid_layer_acc)

        # Compute and save classification statistics
        class_stats = self.compute_layer_classification_stats(valid_loader)
        
        class_counts_filename = filename.replace("all_epochs_layer_weights.csv", "layer_class_counts.csv")
        confusion_filename = filename.replace("all_epochs_layer_weights.csv", "layer_confusion.csv")
        
        # Save class counts
        counts_header = ["epoch", "layer", "total_correct", "total_incorrect"] + [f"class_{c}_correct" for c in range(self.num_class)]
        append_mode = 'a' if os.path.exists(class_counts_filename) else 'w'
        with open(class_counts_filename, append_mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            if append_mode == 'w':
                writer.writerow(counts_header)
            for i in range(len(self.layers)):
                row = [epoch, i, class_stats['per_layer_correct'][i], class_stats['per_layer_incorrect'][i]]
                row.extend(class_stats['per_layer_per_class_correct'][i].tolist())
                writer.writerow(row)
        
        # Save confusion matrices
        confusion_header = ["epoch", "layer", "true_class"] + [f"pred_{c}" for c in range(self.num_class)]
        append_mode = 'a' if os.path.exists(confusion_filename) else 'w'
        with open(confusion_filename, append_mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            if append_mode == 'w':
                writer.writerow(confusion_header)
            for i in range(len(self.layers)):
                for true_c in range(self.num_class):
                    row = [epoch, i, true_c]
                    row.extend(class_stats['per_layer_confusion'][i][true_c].tolist())
                    writer.writerow(row)
        
        # Compute and save cosine similarities between class channel weights
        cos_sim_filename = filename.replace("all_epochs_layer_weights.csv", "channel_cos_sim.csv")
        cos_sim_header = ["epoch", "layer"]
        for class_i in range(self.num_class):
            for class_j in range(class_i + 1, self.num_class):
                cos_sim_header.append(f"cos_sim_{class_i}-{class_j}")
        
        append_mode = 'a' if os.path.exists(cos_sim_filename) else 'w'
        with open(cos_sim_filename, append_mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            if append_mode == 'w':
                writer.writerow(cos_sim_header)
            for i, layer in enumerate(self.layers):
                row = [epoch, i]
                if hasattr(layer, 'current_weights') and layer.current_weights is not None:
                    weights = layer.current_weights
                    cos_sims = []
                    for class_i in range(self.num_class):
                        for class_j in range(class_i + 1, self.num_class):
                            vec_i = weights[:, class_i].unsqueeze(0)
                            vec_j = weights[:, class_j].unsqueeze(0)
                            cos_sim = F.cosine_similarity(vec_i, vec_j).item()
                            cos_sims.append(cos_sim)
                    row.extend(cos_sims)
                else:
                    row.extend([0.0] * (self.num_class * (self.num_class - 1) // 2))
                writer.writerow(row)

    def compute_layer_classification_stats(self, dataloader):
        """Compute classification statistics for each layer on the given dataloader."""
        devices = [next(layer.parameters()).device for layer in self.layers]
        self.eval()
        num_layers = len(self.layers)
        total = 0
        per_layer_correct = [0] * num_layers
        per_layer_per_class_correct = [torch.zeros(self.num_class) for _ in range(num_layers)]
        per_layer_confusion = [torch.zeros(self.num_class, self.num_class) for _ in range(num_layers)]
        with torch.no_grad():
            for x, labels in dataloader:
                shortcut = torch.zeros(1)
                x, labels = x.to(devices[0]), labels.to(devices[0])
                batch_size = labels.size(0)
                total += batch_size
                x = F.layer_norm(x, x.shape[1:])
                for i, layer in enumerate(self.layers):
                    x = x.to(devices[i])
                    labels_dev = labels.to(devices[i])
                    x, g = layer(x)
                    pred = torch.argmax(g, dim=1)
                    correct_mask = (pred == labels_dev)
                    per_layer_correct[i] += correct_mask.sum().item()
                    for c in range(self.num_class):
                        class_mask = (labels_dev == c)
                        per_layer_per_class_correct[i][c] += (correct_mask & class_mask).sum().item()
                        for p in range(self.num_class):
                            pred_mask = (pred == p)
                            per_layer_confusion[i][c][p] += (class_mask & pred_mask).sum().item()
                    # shortcut
                    if self.shortcut_flag[i] == ADD_SHORTCUT:
                        x += shortcut.to(x.device)
                    elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                        x = torch.cat([x, shortcut.to(x.device)], dim=1)
                    if self.input_shortcut_flag[i]:
                        shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()
        per_layer_incorrect = [total - c for c in per_layer_correct]
        return {
            'per_layer_correct': per_layer_correct,
            'per_layer_incorrect': per_layer_incorrect,
            'per_layer_per_class_correct': per_layer_per_class_correct,
            'per_layer_confusion': per_layer_confusion
        }

def test_model(model, test_loader, device):
    model.eval()
    total = 0
    correct = 0
    for x, labels in test_loader:
        x, labels = x.to(device), labels.to(device)
        with torch.no_grad():
            y = model(x)
            pred = torch.argmax(y, dim=1)
            correct += torch.eq(pred, labels).sum().float().item()
        total += labels.size(0)
    test_acc = 100 * correct / total
    return test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DeepForward Training')
    args = argparse.Namespace(
        task='mnist',
        epochs=150,
        train_batch_size=128,
        test_batch_size=128,
        lr=0.08,
        lr_min=0.006,
        weight_decay=5e-4,
        dropout=0.1,
        devices=[0],
        seed=2222,
        task_dir=None,
        arch='resnet',
        parallel=False,
        save_step=5,
        queue_size=5,
        data_dir='./data'
    )

    epochs = args.epochs
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    lr = args.lr
    lr_min = args.lr_min
    weight_decay = args.weight_decay
    dropout = args.dropout

    if args.task_dir is None:
        task_dir = os.path.join('./results/', f'{time.strftime("%Y%m%d-%H%M%S")}_{args.task}_{args.arch}')
    else:
        task_dir = os.path.join('./results/', args.task_dir)

    if task_dir is not None:
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

    in_channels = 1
    num_class = 10
    start_epoch = 0

    root = args.data_dir
    img_size = 32

    devices = []
    for i in args.devices:
        devices.append(torch.device('cuda:0'.format() if torch.cuda.is_available() else 'cpu'))

    seed = 2222
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    checkpoint_path = './checkpoint/checkpoint.pth'
    if task_dir is not None:
        checkpoint_path = os.path.join(task_dir, 'checkpoint.pth')

    task = args.task
    train_loader, valid_loader, test_loader = None, None, None
    if task == 'mnist':
        train_loader, valid_loader, test_loader = get_mnist_dataloader(root=root, train_batch_size=train_batch_size,
                                                                         test_batch_size=test_batch_size, seed=seed)

    model = None
    if args.arch == 'resnet':
        model = Resnet(in_channels=in_channels, num_class=num_class, dropout=dropout,
                       planes=[100,200,400,800], learning_rate=lr, lr_min=lr_min, weight_decay=weight_decay,
                       devices=devices, epochs=epochs)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        start_epoch += 1
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        for i, optimizer in enumerate(model.optimizers):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'][i])
        for i, scheduler in enumerate(model.schedulers):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'][i])
        print(f'Load checkpoint from {checkpoint_path} start from epoch {start_epoch + 1}')

    train_acc, valid_acc, test_acc = 0., 0., 0.
    max_test_acc = 0.
    max_test_acc_pruning = 0.

    weight_filename = os.path.join(task_dir, "all_epochs_layer_weights.csv") if task_dir else "all_epochs_layer_weights.csv"
    stats_filename = os.path.join(task_dir, "layer_weight_stats.csv") if task_dir else "layer_weight_stats.csv"
    class_counts_filename = os.path.join(task_dir, "layer_class_counts.csv") if task_dir else "layer_class_counts.csv"
    confusion_filename = os.path.join(task_dir, "layer_confusion.csv") if task_dir else "layer_confusion.csv"
    cos_sim_filename = os.path.join(task_dir, "channel_cos_sim.csv") if task_dir else "channel_cos_sim.csv"
    info_csv_path = os.path.join(task_dir, f"info_{filename_postfix}.csv") if task_dir else f"info_{filename_postfix}.csv"
    ortho_csv_path = os.path.join(task_dir, f"ortho_metrics_{filename_postfix}.csv") if task_dir else f"ortho_metrics_{filename_postfix}.csv"
    losses_filename = os.path.join(task_dir, f"layer_losses_{filename_postfix}.csv") if task_dir else f"layer_losses_{filename_postfix}.csv"
    weight_dist_filename = os.path.join(task_dir, "weight_dist.csv") if task_dir else "weight_dist.csv"

    # Clear the files if they exist
    for path in [info_csv_path, ortho_csv_path, losses_filename, weight_dist_filename]:
        if os.path.exists(path):
            os.remove(path)

    with open(losses_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = [f'Layer_{i}' for i in range(0, 17)]
        writer.writerow(header)

    # Create the orthogonality CSV
    with open(ortho_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["epoch"] + [f"layer_{i}_ortho_loss" for i in range(17)] + [f"layer_{i}_ortho_score" for i in range(17)]
        writer.writerow(header)

    # Clear the files if they exist
    for file in [weight_filename, stats_filename, class_counts_filename, confusion_filename, cos_sim_filename]:
        if os.path.exists(file):
            os.remove(file)

    for epoch in range(start_epoch, epochs):
        for layer in model.layers:
            layer.update_epoch(epoch)

        if epoch % 10 == 0:
            model.save_layer_weights(weight_filename, epoch=epoch, stats_filename=stats_filename)
            print(f"Appended epoch {epoch} weights to {weight_filename}")
        
        start_time = time.time()
        model.train()
        if args.parallel:
            model.update_pipeline(train_loader, queue_size=args.queue_size, task_dir=task_dir, epoch=epoch)
        else:
            model.update(train_loader, task_dir, epoch)
        model.eval()
        train_time = time.time() - start_time

        # pruning
        pruning_time = time.time()
        valid_acc, before_acc, test_acc, before_test_acc = model.pruning(valid_loader, test_loader, k=5)
        pruning_time = time.time() - pruning_time

        test_time = time.time()
        test_time = time.time() - test_time

        max_test_acc = max(max_test_acc, before_test_acc)
        max_test_acc_pruning = max(max_test_acc_pruning, test_acc)

        model.save_orthogonality_metrics(epoch) 
        
        if epoch % args.save_step == 0:            
            train_acc = test_model(model, train_loader, devices[0])
            end_time = time.time() - start_time

            info = f'Epoch: {(epoch + 1):03d}/{epochs:03d}: ' \
                   f'Train Acc: {train_acc:.2f}% \t || Test training-set Time: {end_time:.2f}s'
            print(info)

            if task_dir is not None:
                with open(os.path.join(task_dir, 'accuracy.csv'), 'a') as f:
                    f.write(f'{epoch},{train_acc},{valid_acc},{test_acc}\n')

        info = f'Epoch: {(epoch + 1):03d}/{epochs:03d}: ' \
                f'Valid Acc: {before_acc:.2f}% -> {valid_acc:.2f}%\t ' \
                f'Test Acc: {before_test_acc:.2f}% -> {test_acc:.2f}%\t || ' \
                f'Train Time: {train_time:.2f}s, ' \
                f'Pruning Time: {pruning_time:.2f}s '\
                f'Test Time: {test_time:.2f}s'\
                f'|| lr: {model.optimizers[0].param_groups[0]["lr"]:.3f}'
        print(info)

        with open(info_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                (model.start_layer).item(),
                (model.end_layer).item(),
                round(before_acc, 2),
                round(valid_acc, 2),
                round(before_test_acc, 2),
                round(test_acc, 2),
                round(train_time, 2),
                round(pruning_time, 2),
                round(test_time, 2),
                round(model.optimizers[0].param_groups[0]["lr"], 3),
                info
            ])

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': [optimizer.state_dict() for optimizer in model.optimizers],
            'scheduler_state_dict': [scheduler.state_dict() for scheduler in model.schedulers],
        }

        if (epoch + 1) % 25 == 0:
            print("==================")
            print(f"until {current_epoch}:")
            print(f"Epoch {(epoch + 1):03d}: Maximum Test Accuracy: {max_test_acc:.2f}%")
            print(f"Epoch {(epoch + 1):03d}: Maximum Test Accuracy Pruning: {max_test_acc_pruning:.2f}%")
            print("==================")

        current_epoch += 1

    start_time = time.time()
    train_acc = test_model(model, train_loader, devices[0])
    end_time = time.time() - start_time

    print(f'Final: Train Acc: {train_acc:.2f}% \t || Test training-set Time: {end_time:.2f}s')
    if task_dir is not None and args.save_step != 1:
        with open(os.path.join(task_dir, 'accuracy.csv'), 'a') as f:
            f.write(f'{epochs},{train_acc},{valid_acc},{test_acc}\n')

    train_layer_acc_list = model.test_local_acc(train_loader)
    test_layer_acc_list = model.test_local_acc(test_loader)
    if task_dir is not None:
        with open(os.path.join(task_dir, 'layer_acc.csv'), 'w') as f:
            for i in range(len(train_layer_acc_list)):
                f.write(f'{train_layer_acc_list[i]},{test_layer_acc_list[i]}\n')

    model = model.to('cpu')
    if task_dir is not None:
        model_path = os.path.join(task_dir, 'model.pth')