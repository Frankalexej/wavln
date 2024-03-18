import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
from sklearn.decomposition import PCA
from scipy.linalg import block_diag
import pickle
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import sem
import argparse

from model_padding import generate_mask_from_lengths_mat, mask_it
from paths import *
from misc_my_utils import *
from model_loss import *
from model_model import VQVAEV1, AEV1
from model_dataset import TargetVowelDatasetBoundary as ThisDataset
from model_dataset import Normalizer, DeNormalizer, TokenMap
from model_dataset import MelSpecTransformDB as TheTransform
from model_dataset import DS_Tools
from reshandler import DictResHandler
from misc_progress_bar import draw_progress_bar
from test_bnd_detect_tools import *
from misc_tools import PathUtils as PU
from misc_tools import AudioCut, ARPABET
from misc_my_utils import time_to_frame

# Constants
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
NUM_LAYERS = 2
EMBEDDING_DIM = 128
REC_SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 64
LOADER_WORKER = 16

############################################ Utils ############################################
def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    # ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest")

def get_endframes(seppos, attn_size): 
    return [0, seppos], [seppos, attn_size]

def create_phoneme_block_matrix(starts, ends, total):
    # Initialize an empty list to store phoneme block matrices
    phoneme_blocks = []
    # Iterate through the phoneme frames to create each block
    for start_frame, end_frame in list(zip(starts, ends))[:-1]:
        num_frames = end_frame - start_frame
        phoneme_block = np.ones((num_frames, num_frames))
        phoneme_blocks.append(phoneme_block)
    num_frames = total - starts[-1]
    phoneme_block = np.ones((num_frames, num_frames))
    phoneme_blocks.append(phoneme_block)
    block_diag_matrix = block_diag(*phoneme_blocks)
    return block_diag_matrix

def post2pre_filter(start, sep, end): 
    return np.block([[np.zeros((sep-start, end))], [np.ones((end-sep, sep-start)), np.zeros((end-sep, end-sep))]])

def biway_filter(start, sep, end): 
    return np.block([[np.zeros((sep-start, sep-start)), np.ones((sep-start, end-sep))], [np.ones((end-sep, sep-start)), np.zeros((end-sep, end-sep))]])

def get_in_phone_attn(attn, starts, ends, total): 
    block_diag_matrix = create_phoneme_block_matrix(starts, ends, total)
    filtered_attn = block_diag_matrix * attn
    in_phoneme_attn = filtered_attn.sum(-1)
    return in_phoneme_attn

def interpolate_traj(current, n_steps=100): 
    current_steps = np.linspace(0, 1, num=len(current))
    target_steps = np.linspace(0, 1, num=n_steps)
    interp_func = interp1d(current_steps, current, kind='linear')
    return interp_func(target_steps)

def cutHid(hid, cutstart, cutend, start_offset=0, end_offset=1): 
    selstart = max(cutstart, int(cutstart + (cutend - cutstart) * start_offset))
    selend = min(cutend, int(cutstart + (cutend - cutstart) * end_offset))
    # hid is (L, H)
    return hid[selstart:selend, :]

# we have very limited data, so we don't need to select, just plot all
def get_toplot(hiddens, sepframes, phi_types, stop_names, offsets=(0, 1)): 
    cutstarts = []
    cutends = []
    for hidden, sepframe, phi_type in zip(hiddens, sepframes, phi_types):
        if phi_type == 'ST':
            cutstarts.append(sepframe)
        else:
            cutstarts.append(0)
        cutends.append(hidden.shape[0])
    
    hid_sel = np.empty((0, 8))
    tag_sel = []
    for (item, start, end, tag) in zip(hiddens, cutstarts, cutends, phi_types): 
        hid = cutHid(item, start, end, offsets[0], offsets[1])
        hidlen = hid.shape[0]
        hid_sel = np.concatenate((hid_sel, hid), axis=0)
        tag_sel += [tag] * hidlen
    return hid_sel, np.array(tag_sel)

def extract_attention_blocks_ST(attention_matrix, sepframe1, sepframe2):
    t_start, t_end = sepframe1, sepframe2
    s_start, s_end = 0, sepframe1
    a_start, a_end = sepframe2, attention_matrix.shape[0]
    # (start, end] for slicing
    # Extracting specific blocks based on the provided indices
    t_to_s = attention_matrix[t_start:t_end, s_start:s_end].sum(axis=1)
    s_to_t = attention_matrix[s_start:s_end, t_start:t_end].sum(axis=1)
    a_to_t = attention_matrix[a_start:a_end, t_start:t_end].sum(axis=1)
    t_to_a = attention_matrix[t_start:t_end, a_start:a_end].sum(axis=1)
    
    return {
        't_to_s': t_to_s,
        's_to_t': s_to_t,
        'a_to_t': a_to_t,
        't_to_a': t_to_a
    }

def extract_attention_blocks_T(attention_matrix, sepframe):
    t_start, t_end = 0, sepframe
    a_start, a_end = sepframe, attention_matrix.shape[0]
    # (start, end] for slicing
    # Extracting specific blocks based on the provided indices
    t_to_a = attention_matrix[t_start:t_end, a_start:a_end].sum(axis=1)
    a_to_t = attention_matrix[a_start:a_end, t_start:t_end].sum(axis=1)
    
    return {
        't_to_a': t_to_a,
        'a_to_t': a_to_t
    }

def plot_attention_trajectory(phi_type, all_attn, all_sepframes1, all_sepframes2, save_path): 
    if phi_type == "ST":
        n_steps = 100
        s_to_t_traj = []
        t_to_s_traj = []
        t_to_a_traj = []
        a_to_t_traj = []
        for i in range(len(all_attn)): 
            this_attn = all_attn[i]
            this_sep_frame1 = all_sepframes1[i]
            this_sep_frame2 = all_sepframes2[i]

            blocks = extract_attention_blocks_ST(this_attn, this_sep_frame1, this_sep_frame2)

            s_to_t_interp = interpolate_traj(blocks['s_to_t'], n_steps)
            t_to_s_interp = interpolate_traj(blocks['t_to_s'], n_steps)
            t_to_a_interp = interpolate_traj(blocks['t_to_a'], n_steps)
            a_to_t_interp = interpolate_traj(blocks['a_to_t'], n_steps)
            s_to_t_traj.append(s_to_t_interp)
            t_to_s_traj.append(t_to_s_interp)
            t_to_a_traj.append(t_to_a_interp)
            a_to_t_traj.append(a_to_t_interp)

        # Convert list of arrays into 2D NumPy arrays for easier manipulation
        group1_array = np.array(s_to_t_traj)
        group2_array = np.array(t_to_s_traj)
        group3_array = np.array(t_to_a_traj)
        group4_array = np.array(a_to_t_traj)

        # Calculate the mean trajectory for each group
        means = np.array([np.mean(group1_array, axis=0), 
                        np.mean(group2_array, axis=0), 
                        np.mean(group3_array, axis=0), 
                        np.mean(group4_array, axis=0)])

        # Calculate the SEM for each step in both groups
        sems = np.array([sem(group1_array, axis=0),
                        sem(group2_array, axis=0),
                        sem(group3_array, axis=0),
                        sem(group4_array, axis=0)])

        # Calculate the 95% CI for both groups
        ci_95s = 1.96 * sems

        # Upper and lower bounds of the 95% CI for both groups
        upper_bounds = means + ci_95s
        lower_bounds = means - ci_95s

        # Plotting
        plt.figure(figsize=(12, 8))
        for mean, upper, lower, label in zip(means, upper_bounds, lower_bounds, ['S-to-P', 'P-to-S', 'P-to-V', 'V-to-P']):
            plt.plot(mean, label=label)
            plt.fill_between(range(n_steps), lower, upper, alpha=0.2)

        plt.xlabel('Normalized Time')
        plt.ylabel('Summed Foreign-Attention')
        plt.title('Comparison of Foreign-Attention Trajectory')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    elif phi_type == "T": 
        n_steps = 100
        t_to_a_traj = []
        a_to_t_traj = []
        for i in range(len(all_attn)): 
            this_attn = all_attn[i]
            this_sep_frame2 = all_sepframes2[i]

            blocks = extract_attention_blocks_T(this_attn, this_sep_frame2)

            t_to_a_interp = interpolate_traj(blocks['t_to_a'], n_steps)
            a_to_t_interp = interpolate_traj(blocks['a_to_t'], n_steps)
            t_to_a_traj.append(t_to_a_interp)
            a_to_t_traj.append(a_to_t_interp)

        # Convert list of arrays into 2D NumPy arrays for easier manipulation
        group3_array = np.array(t_to_a_traj)
        group4_array = np.array(a_to_t_traj)

        # Calculate the mean trajectory for each group
        means = np.array([np.mean(group3_array, axis=0), 
                        np.mean(group4_array, axis=0)])

        # Calculate the SEM for each step in both groups
        sems = np.array([sem(group3_array, axis=0),
                        sem(group4_array, axis=0)])

        # Calculate the 95% CI for both groups
        ci_95s = 1.96 * sems

        # Upper and lower bounds of the 95% CI for both groups
        upper_bounds = means + ci_95s
        lower_bounds = means - ci_95s

        # Plotting
        plt.figure(figsize=(12, 8))
        for mean, upper, lower, label in zip(means, upper_bounds, lower_bounds, ['P-to-V', 'V-to-P']):
            plt.plot(mean, label=label)
            plt.fill_between(range(n_steps), lower, upper, alpha=0.2)

        plt.xlabel('Normalized Time')
        plt.ylabel('Summed Foreign-Attention')
        plt.title('Comparison of Foreign-Attention Trajectory')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

















def get_data(rec_dir, guide_path):
    mytrans = TheTransform(sample_rate=REC_SAMPLE_RATE, 
                        n_fft=N_FFT, n_mels=N_MELS, 
                        normalizer=Normalizer.norm_mvn, 
                        denormalizer=DeNormalizer.norm_mvn)

    st_valid = pd.read_csv(guide_path)

    valid_ds = ThisDataset(rec_dir, 
                        st_valid, 
                        transform=mytrans)

    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=LOADER_WORKER, collate_fn=ThisDataset.collate_fn)
    return valid_loader

def get_data_both(rec_dir, t_guide_path, st_guide_path):
    mytrans = TheTransform(sample_rate=REC_SAMPLE_RATE, 
                        n_fft=N_FFT, n_mels=N_MELS, 
                        normalizer=Normalizer.norm_mvn, 
                        denormalizer=DeNormalizer.norm_mvn)

    st_valid = pd.read_csv(st_guide_path)
    t_valid = pd.read_csv(t_guide_path)
    t_valid["pre_startTime"] = t_valid['stop_startTime']
    all_valid = pd.concat([t_valid, st_valid], ignore_index=True, sort=False)

    valid_ds = ThisDataset(rec_dir, 
                        all_valid, 
                        transform=mytrans)

    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=LOADER_WORKER, collate_fn=ThisDataset.collate_fn)
    return valid_loader


def run_one_epoch(model, single_loader, both_loader, model_save_dir, stop_epoch, res_save_dir): 
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recon_loss = nn.MSELoss(reduction='none')
    masked_recon_loss = MaskedLoss(recon_loss)
    model_loss = masked_recon_loss

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Load model
    model_name = "{}.pt".format(stop_epoch)
    model_path = os.path.join(model_save_dir, model_name)
    state = torch.load(model_path)
    model.load_state_dict(state)
    model.to(device)

    # Run model on data to collect results
    model.eval()
    reshandler = DictResHandler(whole_res_dir=res_save_dir, 
                                 file_prefix=f"all-{stop_epoch}")
    all_ze = []
    all_zq = []
    all_stop_names = []
    all_vowel_names = []
    all_sepframes1 = []
    all_sepframes2 = []
    all_attn = []
    all_recon = []
    all_ori = []
    all_phi_type = []

    for (x, x_lens, pt, sn, vn, sf1, sf2) in both_loader: 
        # name = name[0]

        x_mask = generate_mask_from_lengths_mat(x_lens, device=device)
        
        x = x.to(device)

        x_hat, attn_w, (ze, zq) = model(x, x_lens, x_mask)

        ze = ze.cpu().detach().numpy().squeeze()
        zq = zq.cpu().detach().numpy().squeeze()
        attn_w = attn_w.cpu().detach().numpy().squeeze()
        
        recon_x = x_hat.cpu().detach().numpy().squeeze()
        ori_x = x.cpu().detach().numpy().squeeze()

        all_ze += [ze]
        all_zq += [zq]
        all_attn += [attn_w]
        all_recon += [recon_x]
        all_ori += [ori_x]
        all_stop_names += sn
        all_vowel_names += vn
        all_sepframes1 += sf1
        all_sepframes2 += sf2
        all_phi_type += pt
    
    reshandler.res["ze"] = all_ze
    reshandler.res["zq"] = all_zq
    reshandler.res["sn"] = all_stop_names
    reshandler.res["vn"] = all_vowel_names
    reshandler.res["sep-frame1"] = all_sepframes1
    reshandler.res["sep-frame2"] = all_sepframes2
    reshandler.res["attn"] = all_attn
    reshandler.res["recon"] = all_recon
    reshandler.res["ori"] = all_ori
    reshandler.res["phi-type"] = all_phi_type
    reshandler.save()
    print(f"Results all-{stop_epoch} saved at {res_save_dir}")


    # Plot Reconstructions
    i = 25
    fig, axs = plt.subplots(2, 1)
    plot_spectrogram(all_ori[i].T, title=f"mel-spectrogram of input {all_phi_type[i]}:{all_stop_names[i]}-{all_vowel_names[i]}", ax=axs[0])
    plot_spectrogram(all_recon[i].T, title=f"reconstructed mel-spectrogram {all_phi_type[i]}:{all_stop_names[i]}-{all_vowel_names[i]}", ax=axs[1])
    fig.tight_layout()
    plt.savefig(os.path.join(res_save_dir, f"recon-at-{stop_epoch}.png"))
    plt.close()

    for selector in ["ST", "T"]:
        selected_tuples = [(sf1, sf2, attn) for pt, sf1, sf2, attn in zip(all_phi_type,  
                                                          all_sepframes1, 
                                                          all_sepframes2, 
                                                          all_attn) if pt == selector]
        selected_sf1s, selected_sf2s, selected_attns = zip(*selected_tuples)

        # Plot Attention Trajectory
        plot_attention_trajectory(selector, selected_attns, selected_sf1s, selected_sf2s, os.path.join(res_save_dir, f"attntraj-{selector}-at-{stop_epoch}.png"))
    return 0



def main(train_name, ts, run_number, model_type, model_save_dir, res_save_dir, guide_dir): 
    # Dirs
    rec_dir = train_cut_phone_
    # Check model path
    assert PU.path_exist(model_save_dir)
    assert PU.path_exist(guide_dir)

    # Load data
    st_guide_path = os.path.join(guide_dir, "ST-valid.csv")
    single_loader = get_data(rec_dir, st_guide_path)
    # note that we always use the balanced data to evaluate, this is because we want the evaluation to have 
    # equal contrast, instead of having huge bias. 
    both_loader = get_data_both(rec_dir, os.path.join(guide_dir, "T-valid-sampled.csv"), st_guide_path)

    if model_type == "ae":
        model = AEV1(enc_size_list=ENC_SIZE_LIST, 
                   dec_size_list=DEC_SIZE_LIST, 
                   embedding_dim=EMBEDDING_DIM, 
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
    elif model_type == "vqvae":
        model = VQVAEV1(enc_size_list=ENC_SIZE_LIST, 
                   dec_size_list=DEC_SIZE_LIST, 
                   embedding_dim=EMBEDDING_DIM, 
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
    else:
        model = VQVAEV1(enc_size_list=ENC_SIZE_LIST, 
                   dec_size_list=DEC_SIZE_LIST, 
                   embedding_dim=EMBEDDING_DIM, 
                   num_layers=NUM_LAYERS, dropout=DROPOUT)

    # sil_list = []
    for epoch in range(0, 100): 
        run_one_epoch(model, single_loader, both_loader, model_save_dir, epoch, res_save_dir)
    return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--timestamp', '-ts', type=str, default="0000000000", help="Timestamp for project, better be generated by bash")
    parser.add_argument('--runnumber', '-rn', type=str, default="0", help="Run number")
    parser.add_argument('--gpu', '-gpu', type=int, default=0, help="Choose the GPU to work on")
    parser.add_argument('--model','-m',type=str, default = "ae",help="Model type: ae or vqvae")
    parser.add_argument('--condition','-cd',type=str, default="b", help='Condition: b (balanced), u (unbalanced), nt (no-T)')
    args = parser.parse_args()

    # set device number
    torch.cuda.set_device(args.gpu)

    ts = args.timestamp # this timestamp does not contain run number
    rn = args.runnumber
    train_name = "C_0B"
    if not PU.path_exist(os.path.join(model_save_, f"{train_name}-{ts}-{rn}")):
        raise Exception(f"Training {train_name}-{ts}-{rn} does not exist! ")
    
    model_save_dir = os.path.join(model_save_, f"{train_name}-{ts}-{rn}", args.model, args.condition)
    guide_dir = os.path.join(model_save_, f"{train_name}-{ts}-{rn}", "guides")
    res_save_dir = os.path.join(model_save_, f"eval-{train_name}-{ts}")
    this_model_condition_dir = os.path.join(res_save_dir, args.model, args.condition, f"{rn}")
    mk(this_model_condition_dir)

    main(train_name, ts, args.runnumber, args.model, model_save_dir, this_model_condition_dir, guide_dir)