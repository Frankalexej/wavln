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
from model_dataset import TargetDatasetBoundary as ThisDataset
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

def plot_attention_trajectory(all_attn, all_stop_names, all_sepframes, save_path): 
    n_steps = 100
    first_traj = []
    second_traj = []
    for i in range(len(all_attn)): 
        this_attn = all_attn[i]
        this_sn = all_stop_names[i]
        this_sepposition = all_sepframes[i]
        attn_size = this_attn.shape[0]

        this_biway_attn_filter = biway_filter(0, this_sepposition, attn_size)
        filtered_attn = this_biway_attn_filter * this_attn
        summed_filtered_attn = filtered_attn.sum(-1)

        first_interp = interpolate_traj(summed_filtered_attn[:this_sepposition], n_steps)
        second_interp = interpolate_traj(summed_filtered_attn[this_sepposition:], n_steps)[::-1]
        first_traj.append(first_interp)
        second_traj.append(second_interp)

    # Convert list of arrays into 2D NumPy arrays for easier manipulation
    group1_array = np.array(first_traj)
    group2_array = np.array(second_traj)[::-1]

    # Calculate the mean trajectory for each group
    mean_trajectory_group1 = np.mean(group1_array, axis=0)
    mean_trajectory_group2 = np.mean(group2_array, axis=0)

    # Calculate the SEM for each step in both groups
    sem_group1 = sem(group1_array, axis=0)
    sem_group2 = sem(group2_array, axis=0)

    # Calculate the 95% CI for both groups
    ci_95_group1 = 1.96 * sem_group1
    ci_95_group2 = 1.96 * sem_group2

    # Upper and lower bounds of the 95% CI for both groups
    upper_bound_group1 = mean_trajectory_group1 + ci_95_group1
    lower_bound_group1 = mean_trajectory_group1 - ci_95_group1
    upper_bound_group2 = mean_trajectory_group2 + ci_95_group2
    lower_bound_group2 = mean_trajectory_group2 - ci_95_group2

    # Plotting
    plt.figure(figsize=(12, 8))
    # Mean trajectory for Group 1
    plt.plot(mean_trajectory_group1, label='/s/', color='blue')
    # 95% CI area for Group 1
    plt.fill_between(range(n_steps), lower_bound_group1, upper_bound_group1, color='blue', alpha=0.2)
    # Mean trajectory for Group 2
    plt.plot(mean_trajectory_group2, label='Stop', color='red')
    # 95% CI area for Group 2
    plt.fill_between(range(n_steps), lower_bound_group2, upper_bound_group2, color='red', alpha=0.2)

    plt.xlabel('Normalized Distance from Boundary')
    plt.ylabel('Summed Foreign-Attention')
    plt.title('Comparison of Foreign-Attention Trajectory')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_silhouette(silarray, save_path): 
    n_steps = 100
    # Convert list of arrays into 2D NumPy arrays for easier manipulation
    group1_array = np.array(silarray)

    # Calculate the mean trajectory for each group
    mean_trajectory_group1 = np.mean(group1_array, axis=0)

    # Calculate the SEM for each step in both groups
    sem_group1 = sem(group1_array, axis=0)

    # Calculate the 95% CI for both groups
    ci_95_group1 = 1.96 * sem_group1

    # Upper and lower bounds of the 95% CI for both groups
    upper_bound_group1 = mean_trajectory_group1 + ci_95_group1
    lower_bound_group1 = mean_trajectory_group1 - ci_95_group1

    # Plotting
    plt.figure(figsize=(12, 8))
    # Mean trajectory for Group 1
    plt.plot(mean_trajectory_group1, label='silhouette', color='blue')
    # 95% CI area for Group 1
    plt.fill_between(range(n_steps), lower_bound_group1, upper_bound_group1, color='blue', alpha=0.2)

    plt.xlabel('Epoch')
    plt.ylabel('Silhouette Score (40%~60%)')
    plt.title('Silhouette Score Across Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_attention_statistics(all_attn, all_sepframes, save_path): 
    all_boundary_attn = []
    all_middle_attn = []
    all_sib_middle_attn = []

    for i in range(len(all_attn)): 
        this_attn = all_attn[i]
        this_sepposition = all_sepframes[i]
        attn_size = this_attn.shape[0]
        this_startframes, this_endframes = get_endframes(this_sepposition, attn_size)

        boundary_positions = this_endframes[:-1]
        middle_positions = ((np.array(this_startframes) + np.array(this_endframes)) / 2).astype(int).tolist()[:-1]
        sib_middle_positions = ((np.array(this_startframes) + np.array(this_endframes)) / 2).astype(int).tolist()[1:]

        if len(boundary_positions) < 1: 
            continue

        inphone_attn = get_in_phone_attn(this_attn, this_startframes, this_endframes, attn_size)

        for pos in boundary_positions: 
            all_boundary_attn += inphone_attn[boundary_positions].tolist()
        for pos in middle_positions: 
            all_middle_attn += inphone_attn[middle_positions].tolist()
        for pos in sib_middle_positions: 
            all_sib_middle_attn += inphone_attn[sib_middle_positions].tolist()

    # Sample data for three groups
    group1 = all_boundary_attn  # Assuming this is defined elsewhere
    group2 = all_middle_attn    # Assuming this is defined elsewhere
    group3 = all_sib_middle_attn    # You'll need to define this

    # Calculate means for each group
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    mean3 = np.mean(group3)  # Mean for the third group

    # Calculate standard error of the mean (SEM) for each group
    sem1 = np.std(group1, ddof=1) / np.sqrt(len(group1))
    sem2 = np.std(group2, ddof=1) / np.sqrt(len(group2))
    sem3 = np.std(group3, ddof=1) / np.sqrt(len(group3))  # SEM for the third group

    # Prepare plot details
    labels = ['Boundary', 'T-Middle', 'S-Middle']  # Include a label for the third group
    means = [mean1, mean2, mean3]  # Include mean for the third group
    errors = [sem1, sem2, sem3]  # Include SEM for the third group

    # Create the bar plot
    plt.bar(labels, means, yerr=errors, capsize=5, color=['blue', 'orange', 'green'], alpha=0.75)

    # Add title and labels to the plot
    plt.title('In-phone Attention')
    plt.ylabel('Attention Weight')
    plt.savefig(save_path)
    plt.close()



def read_result_at(res_save_dir, epoch): 
    all_handler = DictResHandler(whole_res_dir=res_save_dir, 
                                 file_prefix=f"all-{epoch}")

    hidrep_handler = DictResHandler(whole_res_dir=res_save_dir, 
                                 file_prefix=f"hidrep-{epoch}")

    all_handler.read()
    hidrep_handler.read()

    return all_handler.res, hidrep_handler.res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--timestamp', '-ts', type=str, default="0000000000", help="Timestamp for project, better be generated by bash")
    parser.add_argument('--gpu', '-gpu', type=int, default=0, help="Choose the GPU to work on")
    parser.add_argument('--model','-m',type=str, default = "ae",help="Model type: ae or vqvae")
    parser.add_argument('--condition','-cd',type=str, default="b", help='Condition: b (balanced), u (unbalanced), nt (no-T)')
    args = parser.parse_args()

    # set device number
    torch.cuda.set_device(args.gpu)

    ts = args.timestamp # this timestamp does not contain run number
    train_name = "C_0A"
    res_save_dir = os.path.join(model_save_, f"eval-{train_name}-{ts}")
    model_condition_dir = os.path.join(res_save_dir, args.model, args.condition)
    print(f"Model condition dir: {model_condition_dir}")
    assert PU.path_exist(model_condition_dir)
    this_save_dir = os.path.join(model_condition_dir, "integrated_results")
    mk(this_save_dir)

    sil_list = []
    for run_number in range(1, 11):
        this_model_condition_dir = os.path.join(model_condition_dir, f"{run_number}")
        sillist_handler = DictResHandler(whole_res_dir=this_model_condition_dir, 
                                    file_prefix=f"silscore")

        sillist_handler.read()
        sil_list.append(sillist_handler.res["sillist"])
    plot_silhouette(sil_list, os.path.join(this_save_dir, "silhouette.png"))

    for epoch in range(0, 100): 
        cat_attns = []
        cat_sepframes = []
        cat_stop_names = []

        for run_number in range(1, 11):
            this_model_condition_dir = os.path.join(model_condition_dir, f"{run_number}")
            allres, hidrepres = read_result_at(this_model_condition_dir, epoch)
            cat_attns += allres["attn"]
            cat_sepframes += allres["sep-frame"]
            cat_stop_names += allres["sn"]
        plot_attention_trajectory(cat_attns, cat_stop_names, cat_sepframes, os.path.join(this_save_dir, f"attntraj-at-{epoch}.png"))
        plot_attention_statistics(cat_attns, cat_sepframes, os.path.join(this_save_dir, f"attnstat-at-{epoch}.png"))