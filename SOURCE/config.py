#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch

# FILES INFO
DATA_DIR = os.path.join("../../DATA")
NUMPY_DIR = os.path.join(DATA_DIR, "NUMPY")
RESULT_DIR = os.path.join(DATA_DIR, "RESULT")
MODEL_DIR = os.path.join(DATA_DIR, "MODEL")

# TIME SERIES INFO
train_year = {"start":1980, "end":2000} 
validation_year = {"start":2000, "end":2005}
test_year = {"start":2005, "end":2015}
window = 365

# CHANNELS INFO
channels = list(range(33))
static_channels = channels[:27]
weather_channels = channels[27:32]
sf_channels = [channels[-1]]

# LABELS INFO
add = 0.005
classes = 1
unknown = -999

# TRAIN INFO
gpu = "cuda"
device = torch.device(gpu)
code_dim = 32
n_clusters = 20
epochs = 200
batch_size = 200
learning_rate = 0.005
alpha = 1.0

# MODEL INFO
recon_weight = 0
static_weight = 1.0
triplet_weight = 0