# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:23:30 2019

@author: Mir

A simple CNN for Fashion-MNIST dataset.
"""

# Importing neccessary modules
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import pandas as pd

# Torch options
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

# Definitions
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):

        # (2) hidden conv layer
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.fc1(t.reshape(-1, 12 * 4 * 4)))
        t = F.relu(self.fc2(t))

        # (6) output layer
        t = self.out(t)

        return t


def get_num_correct(preds, lables):
    return preds.argmax(dim=1).eq(lables).sum().item()


def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    
    for batch in loader:
        
        images, labels = batch
        
        preds = model(images)
        
        #print(preds.shape)
        
        all_preds = torch.cat(
                (all_preds, preds),
                dim=0
        )
        
        #print(all_preds.shape)
        
    return all_preds


class RunBuilder():
    @staticmethod
    def get_runs(params):
        
        Run = namedtuple('Run', params.keys())
        
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
            
        return runs

class RunManager():
    
    def __init__(self):
        
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = 0
        
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = 0
        
        self.network = None
        self.loader = None
        self.tb = None
        
    def begin_run(self, run, network, loader):
        
        self.run_start_time = time.time()
        
        self.run_params = run
        self.run_count += 1
        
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')
        
        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)
        
        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)
        
    def end_run(self):
        
        self.tb.close()
        self.epoch_count = 0
        
    def beign_epoch(self):
        
        self.epoch_start_time = time.time()
        
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        
    def end_epoch(self):
        
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)
        
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        
        for name, param in self.network.named_parameters():
            
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
            
        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        
    def track_loss(self, loss):
        
        self.epoch_loss += loss.item() * self.loader.batch_size
    
    def track_num_correct(self, preds, labels):
        
        self.epoch_num_correct += get_num_correct(preds, labels)
    
    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    
    def save(self, file_name):
        
        pd.DataFrame.from_dict(
                    self.run_data,
                    orient='columns'
                ).to_csv(f'{file_name}.csv')
        
        with open(f'{file_name}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

#############################################################################
    
if __name__ == '__main__':

    num_batch = 100
    num_epoch = 5
    
    params = dict(
    lr = [.01, .001],
    batch_size = [10, 100, 1000],
    shuffle = [True, False]
    )
    
    # Reading dataset
    train_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    
    m = RunManager()
    for run in RunBuilder.get_runs(params):
        
        network = Network()
        loader = torch.utils.data.DataLoader(train_set, run.batch_size)
        optimizer = optim.Adam(network.parameters(), lr=run.lr)
    
        # Training stage
        m.begin_run(run, network, loader)
        for epoch in range(num_epoch):
        
            m.beign_epoch()
        
            for batch in loader:
        
                images = batch[0]
                labels = batch[1]
        
                pred = network(images)
                loss = F.cross_entropy(pred, labels)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                total_loss += loss.item()
                total_correct += get_num_correct(pred, labels)
                
                m.track_loss(loss)
                m.track_num_correct(pred, labels)
                
            m.end_epoch()
            
        m.end_run()
        
    m.save('results')
        
#            print("epoch:", epoch, "total correct: ", total_correct, "loss:",
#                  total_loss)
        
#        # Prediction stage
#        with torch.no_grad():
#            prediction_loader = torch.utils.data.DataLoader(train_set,
#                                                            batch_size=10000)
#            train_preds = get_all_preds(network, prediction_loader)
#        
#        
#        preds_correct = get_num_correct(train_preds, train_set.targets)
#        
#        print('total correct:', preds_correct)
#        print('accuracy:', preds_correct / len(train_set))

#sample = next(iter(train_set))
#image, label = sample
#
#pred = network(image.unsqueeze(0))
#print(pred)

