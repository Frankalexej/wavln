# AI Lab Run Reconstruction + Boundary Detection + HidDim Investigation

import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import random
from IPython.display import Audio
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.manifold import TSNE   # one type of clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from itertools import combinations
import argparse

from model_padding import generate_mask_from_lengths_mat, mask_it
from paths import *
from misc_my_utils import *
from model_loss import *
from model_model import SimplerPhxLearnerInit as TheLearner
from model_dataset import WordDatasetPath
from model_dataset import Normalizer, DeNormalizer
from model_dataset import MelSpecTransformDB as TheTransform
from model_dataset import DS_Tools
from reshandler import WordEncodeResHandler
from misc_progress_bar import draw_progress_bar
from test_bnd_detect_tools import *
from misc_tools import PathUtils as PU
from misc_tools import AudioCut, ARPABET

def inferHiddim(model, loader, stop_epoch): 
    model.eval()
    reshandler = WordEncodeResHandler(whole_res_dir=model_save_dir, 
                                 file_prefix=f"encode-{stop_epoch}")
    all_res = []
    all_name = []

    for (x, x_lens, name) in tqdm(loader): 
        name = name[0]

        x_mask = generate_mask_from_lengths_mat(x_lens, device=device)
        
        x = x.to(device)

        hid_r = model.encode(x, x_lens, x_mask)

        hid_r = hid_r.cpu().detach().numpy().squeeze()

        all_res += [hid_r]
        # note that this is bit different, not each frame, but each sequence is treated as one data point
        all_name += [name]
    

    reshandler.res = all_res
    reshandler.name = all_name
    reshandler.save()
    return reshandler




if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--ts', '-ts', type=str, default="0000000000", help="Timestamp")
    parser.add_argument('--stop_epoch', '-stop_epoch', type=str, default="0", help="Selected Epoch")
    args = parser.parse_args()

    rec_dir = train_cut_word_
    train_guide_path = os.path.join(src_, "guide_train.csv")
    valid_guide_path = os.path.join(src_, "guide_validation.csv")
    test_guide_path = os.path.join(src_, "guide_test.csv")

    EPOCHS = 10
    BATCH_SIZE = 1

    INPUT_DIM = 64
    OUTPUT_DIM = 64

    INTER_DIM_0 = 32
    INTER_DIM_1 = 16
    INTER_DIM_2 = 8

    ENC_SIZE_LIST = [INPUT_DIM, INTER_DIM_0, INTER_DIM_1, INTER_DIM_2]
    DEC_SIZE_LIST = [OUTPUT_DIM, INTER_DIM_0, INTER_DIM_1, INTER_DIM_2]

    DROPOUT = 0.5

    REC_SAMPLE_RATE = 16000
    N_FFT = 400
    N_MELS = 64

    LOADER_WORKER = 28

    ts = args.ts
    stop_epoch = args.stop_epoch
    train_name = "A_01"
    model_save_dir = os.path.join(model_save_, f"{train_name}-{ts}")
    assert PU.path_exist(model_save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recon_loss = nn.MSELoss(reduction='none')
    masked_recon_loss = MaskedLoss(recon_loss)
    model_loss = masked_recon_loss

    model = TheLearner(enc_size_list=ENC_SIZE_LIST, dec_size_list=DEC_SIZE_LIST, num_layers=2)
    # model = TheLearner(enc_size_list=ENC_SIZE_LIST, dec_size_list=DEC_SIZE_LIST, num_layers=1)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model_name = "{}.pt".format(stop_epoch)
    model_path = os.path.join(model_save_dir, model_name)
    state = torch.load(model_path)

    model.load_state_dict(state)
    model.to(device)



