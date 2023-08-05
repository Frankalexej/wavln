import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
# import csv
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure
import pickle
from paths import *
from my_utils import *
from padding import generate_mask_from_lengths_mat, mask_it, masked_loss
from datetime import datetime

from model import PhonLearn_Net
from mydataset import *


model_save_dir = model_eng_save_dir
random_log_path = word_seg_anno_log_path
random_path = word_seg_anno_path
anno_log_path = phone_seg_anno_path


