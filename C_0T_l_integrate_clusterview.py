"""
The Hopkins statistic is specifically designed to measure the clustering tendency of a dataset. 
It evaluates whether the data is significantly different from a uniformly random distribution. 
A high Hopkins value (close to 1) suggests that the data has a strong tendency to form clusters, 
while a value around 0.5 indicates that the data is uniformly distributed without any clustering structure. 
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from C_0X_defs import *
from C_0Y_evaldefs import *
import itertools
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score


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
    selstart = max(cutstart, math.floor(cutstart + (cutend - cutstart) * start_offset))
    selend = min(cutend, math.ceil(cutstart + (cutend - cutstart) * end_offset))
    # hid is (L, H)
    return hid[selstart:selend, :]

def separate_and_sample_data(data_array, tag_array, sample_size):
    # Ensure data_array and tag_array are numpy arrays
    data_array = np.array(data_array)
    tag_array = np.array(tag_array)
    tags = np.unique(tag_array)
    data_list = []
    tag_list = []
    for tag in tags: 
        filtered_data, filtered_tag = filter_data_by_tags(data_array, tag_array, [tag])
        indices = np.random.choice(len(filtered_data), size=sample_size, replace=(sample_size > len(filtered_data)))
        selected_data = filtered_data[indices]
        selected_tag = filtered_tag[indices]
        data_list.append(selected_data)
        tag_list.append(selected_tag)
    return data_list, tag_list

# we have very limited data, so we don't need to select, just plot all
def get_toplot(hiddens, sepframes1, sepframes2, phi_types, stop_names, offsets=(0, 1), contrast_in="asp", merge=True, hidden_dim=8): 
    # collect the start and end frames for each phoneme
    cutstarts = []
    cutends = []
    for sepframe1, sepframe2, phi_type in zip(sepframes1, sepframes2, phi_types):
        # if phi_type == 'ST':
        #     cutstarts.append(sepframe1)
        # else:
        #     cutstarts.append(0)
        # we deleted the above code because now we have SIL. 
        cutstarts.append(sepframe1)
        cutends.append(sepframe2)

    if contrast_in == "asp": 
        tags_list = phi_types
    elif contrast_in == "stop":
        tags_list = stop_names
    else:
        raise ValueError("Contrast_in must be one of 'asp' or 'stop'")
    
    hid_sel = np.empty((0, hidden_dim))
    tag_sel = []
    for (item, start, end, tag) in zip(hiddens, cutstarts, cutends, tags_list): 
        hid = cutHid(item, start, end, offsets[0], offsets[1])
        if merge:
            hid = np.mean(hid, axis=0, keepdims=True)
            hid_sel = np.concatenate((hid_sel, hid), axis=0)
            tag_sel += [tag]
        else: 
            hidlen = hid.shape[0]
            hid_sel = np.concatenate((hid_sel, hid), axis=0)
            tag_sel += [tag] * hidlen
    return hid_sel, np.array(tag_sel)

def plot_silhouette(silarray_1, silarray_2, save_path): 
    # Convert list of arrays into 2D NumPy arrays for easier manipulation
    # group1_array = np.array(silarray_1)
    # group2_array = np.array(silarray_2)

    group1_array = silarray_1
    group2_array = silarray_2

    n_steps = group1_array.shape[1]
    assert n_steps == group2_array.shape[1]

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
    plt.plot(mean_trajectory_group1, label='Aspiration', color='blue')
    # 95% CI area for Group 1
    plt.fill_between(range(n_steps), lower_bound_group1, upper_bound_group1, color='blue', alpha=0.2)
    # Mean trajectory for Group 2
    plt.plot(mean_trajectory_group2, label='Place', color='red')
    # 95% CI area for Group 2
    plt.fill_between(range(n_steps), lower_bound_group2, upper_bound_group2, color='red', alpha=0.2)

    plt.xlabel('Epoch')
    plt.ylabel('ARI')
    plt.title('ARI Across Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_hopkins(array, save_path): 
    # array: (epoch, run)

    n_steps = array.shape[0]

    # Calculate the mean trajectory for each group
    mean_hopkins = np.mean(array, axis=1)

    # Calculate the SEM for each step in both groups
    sem_hopkins = sem(array, axis=1)

    # Calculate the 95% CI for both groups
    ci_95_hopkins = 1.96 * sem_hopkins

    # Upper and lower bounds of the 95% CI for both groups
    upper_bound_hopkins = mean_hopkins + ci_95_hopkins
    lower_bound_hopkins = mean_hopkins - ci_95_hopkins

    # Plotting
    plt.figure(figsize=(12, 8))
    # Mean trajectory for Group 1
    plt.plot(mean_hopkins, label='Segment', color='blue')
    # 95% CI area for Group 1
    plt.fill_between(range(n_steps), lower_bound_hopkins, upper_bound_hopkins, color='blue', alpha=0.2)

    plt.xlabel('Epoch')
    plt.ylabel('Hopkins Statistic')
    plt.title('Hopkins Statistic Across Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_elbows(array, save_path, title="Elbow Method For Optimal k"): 
    # array: (run, k)
    n_steps = array.shape[1]
    mean_run = np.mean(array, axis=0) # mean over runs of each k
    sem_run = sem(array, axis=0)
    ci_95_run = 1.96 * sem_run
    upper_bound_run = mean_run + ci_95_run
    lower_bound_run = mean_run - ci_95_run

    # Plotting
    plt.figure(figsize=(12, 8))
    # Mean trajectory
    plt.plot(range(n_steps), mean_run, 'bo-', markersize=8)
    # 95% CI area
    plt.fill_between(range(n_steps), lower_bound_run, upper_bound_run, alpha=0.2)

    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title(title)
    # plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def hopkins_statistic(X, n_samples=100):
    """
    Calculate the Hopkins statistic for the dataset X.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
        The data matrix to be tested.
    n_samples : int, optional (default=100)
        Number of samples to generate for the test.
        
    Returns:
    hopkins_stat : float
        The Hopkins statistic, ranging from 0 to 1.
    """
    X = np.array(X)
    n, d = X.shape
    
    if n_samples > n:
        n_samples = n
        
    # Randomly select n_samples points from the dataset
    random_indices = np.random.choice(range(n), n_samples, replace=False)
    X_sample = X[random_indices]

    # Generate n_samples random points from the uniform distribution over the range of X
    X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
    X_uniform_random = np.random.uniform(X_min, X_max, (n_samples, d))

    # Compute nearest neighbors for both real and random points
    nbrs_real = NearestNeighbors(n_neighbors=2).fit(X)
    nbrs_random = NearestNeighbors(n_neighbors=1).fit(X)

    u_distances, _ = nbrs_random.kneighbors(X_uniform_random, n_neighbors=1)
    w_distances, _ = nbrs_real.kneighbors(X_sample, n_neighbors=2)

    # Hopkins statistic
    hopkins_stat = np.sum(u_distances) / (np.sum(u_distances) + np.sum(w_distances[:, 1]))

    return hopkins_stat

def plot_clusterview(data, ground_truth_labels, cluster_labels, save_path, title="Clusters with Ground Truth Classes"): 
    # Reduce dimensions if needed
    reduced_data = PCA(n_components=2).fit_transform(data)

    # Plot ground truth
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(ground_truth_labels)

    for i, label in enumerate(unique_labels):
        plt.scatter(reduced_data[ground_truth_labels == label, 0], 
                    reduced_data[ground_truth_labels == label, 1],
                    label=f'Class {label}',
                    marker='o', edgecolor='black', s=100, alpha=0.5)

    # Overlay cluster labels
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', marker='x')

    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--timestamp', '-ts', type=str, default="0000000000", help="Timestamp for project, better be generated by bash")
    parser.add_argument('--gpu', '-gpu', type=int, default=0, help="Choose the GPU to work on")
    parser.add_argument('--model','-m',type=str, default = "ae",help="Model type: ae or vqvae")
    parser.add_argument('--condition','-cd',type=str, default="b", help='Condition: b (balanced), u (unbalanced), nt (no-T)')
    parser.add_argument('--zlevel','-zl',type=str, default="hidrep", help='hidrep / attnout')
    args = parser.parse_args()

    test_name = "clusterview_cluster2"

    # set device number
    torch.cuda.set_device(args.gpu)
    ts = args.timestamp # this timestamp does not contain run number
    train_name = "C_0T"
    res_save_dir = os.path.join(model_save_, f"eval-{train_name}-{ts}")

    model_type = args.model
    model_condition = args.condition
    zlevel = args.zlevel
    model_condition_dir = os.path.join(res_save_dir, model_type, model_condition)
    print(model_condition_dir)
    assert PU.path_exist(model_condition_dir)
    this_save_dir = os.path.join(model_condition_dir, "integrated_results")
    mk(this_save_dir)
    the_good_saving_dir_model_condition = os.path.join(res_save_dir, test_name, model_condition, model_type)
    mk(the_good_saving_dir_model_condition)

    if model_type == "recon4-phi": 
        hidden_dim = 4
    elif model_type == "recon8-phi": 
        hidden_dim = 8
    elif model_type == "recon16-phi": 
        hidden_dim = 16
    elif model_type == "recon32-phi": 
        hidden_dim = 32
    else: 
        raise ValueError("Model type must be one of 'recon3-phi', 'recon8-phi', 'recon32-phi', 'recon100-phi'")
    learned_runs = [1, 2, 3, 4, 5]
    string_learned_runs = [str(num) for num in learned_runs]
    strseq_learned_runs = "".join(string_learned_runs)

    stop_list_epochs = [] # list for each epoch of lists of sse for each run
    asp_list_epochs = []
    for epoch in range(0, 100): 
        # 先循环epoch，再循环run
        stop_list_runs = []
        asp_list_runs = []
        print(f"Processing {model_type} in epoch {epoch}...")
        plot_run = random.choice(learned_runs)
        for run_number in learned_runs:
            this_model_condition_dir = os.path.join(model_condition_dir, f"{run_number}")
            hidrep_handler = DictResHandler(whole_res_dir=this_model_condition_dir, 
                                file_prefix=f"all-{epoch}")
            hidrep_handler.read()
            hidrep = hidrep_handler.res
            if zlevel == "hidrep": 
                all_zq = hidrep["ze"]
            elif zlevel == "attnout": 
                all_zq = hidrep["zq"]
            else: 
                raise ValueError("zlevel must be one of 'hidrep' or 'attnout'")
            
            all_sepframes1 = hidrep["sep-frame1"]
            all_sepframes2 = hidrep["sep-frame2"]
            all_phi_type = hidrep["phi-type"]
            all_stop_names = hidrep["sn"]

            # Silhouette Score
            cluster_groups = ["P", "T", "K"]
            hidr_cs, tags_cs = get_toplot(hiddens=all_zq, 
                                            sepframes1=all_sepframes1,
                                            sepframes2=all_sepframes2,
                                            phi_types=all_phi_type,
                                            stop_names=all_stop_names,
                                            offsets=(0, 1), 
                                            contrast_in="stop", 
                                            merge=True, 
                                            hidden_dim=hidden_dim)
            color_translate = {item: idx for idx, item in enumerate(cluster_groups)}
            hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)
            # Get clusters
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(hidr_cs)
            cluster_labels = kmeans.labels_
            # Calculate ARI
            ari = adjusted_rand_score(tags_cs, cluster_labels)
            stop_list_runs.append(ari)
            # Plot View
            if run_number == plot_run: 
                plot_clusterview(data=hidr_cs, 
                                ground_truth_labels=tags_cs, 
                                cluster_labels=cluster_labels, 
                                save_path=os.path.join(the_good_saving_dir_model_condition, f"01-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}@{epoch}.png"), 
                                title=f"Clusters and Ground Truth {model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}@{epoch}")

            # aspiration contrast
            cluster_groups = ["T", "ST"]
            hidr_cs, tags_cs = get_toplot(hiddens=all_zq, 
                                            sepframes1=all_sepframes1,
                                            sepframes2=all_sepframes2,
                                            phi_types=all_phi_type,
                                            stop_names=all_stop_names,
                                            offsets=(0, 1), 
                                            contrast_in="asp", 
                                            merge=True, 
                                            hidden_dim=hidden_dim)
            color_translate = {item: idx for idx, item in enumerate(cluster_groups)}
            hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)
            # Get clusters
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(hidr_cs)
            cluster_labels = kmeans.labels_
            # Calculate ARI
            ari = adjusted_rand_score(tags_cs, cluster_labels)
            asp_list_runs.append(ari)
            # Plot View
            if run_number == plot_run: 
                plot_clusterview(data=hidr_cs, 
                                ground_truth_labels=tags_cs, 
                                cluster_labels=cluster_labels, 
                                save_path=os.path.join(the_good_saving_dir_model_condition, f"02-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}@{epoch}.png"), 
                                title=f"Clusters and Ground Truth {model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}@{epoch}")

        stop_list_epochs.append(stop_list_runs) # 把每一个epoch的结果汇总，因为最后我们要保存结果，跑起来挺费时间的
        asp_list_epochs.append(asp_list_runs)

    stop_list_epochs = np.array(stop_list_epochs)
    asp_list_epochs = np.array(asp_list_epochs)
    stop_list_epochs = stop_list_epochs.transpose(1, 0)
    asp_list_epochs = asp_list_epochs.transpose(1, 0)
    plot_silhouette(asp_list_epochs, stop_list_epochs, os.path.join(res_save_dir, test_name, f"03-ari-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"))
    np.save(os.path.join(res_save_dir, test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), stop_list_epochs)
    np.save(os.path.join(res_save_dir, test_name, f"05-save-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), asp_list_epochs)
    print("Done.")