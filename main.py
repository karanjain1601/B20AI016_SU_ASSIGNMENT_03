import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq

from model import Model
from dataset import Custom_Dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(device = device)

DATASET_PATH = r"C:\Users\kjain\Speech_Understanding\SU_PA3\Dataset_Speech_Assignment\Dataset_Speech_Assignment"
dataloader = Custom_Dataloader(DATASET_PATH, batch_size=32, shuffle=True)


