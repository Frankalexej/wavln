"""
In this script, we will use vowels but stop consonant tags to evaluate the model. 
This is because stops are short and listeners usually use acoustic cues from the vowel 
to identify the stop consonant. We will check whether the model also did the same. 
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from C_0X_defs import *
from C_0Y_evaldefs import *


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
    if cutend is None: 
        cutend = hid.shape[0]
    selstart = max(cutstart, math.floor(cutstart + (cutend - cutstart) * start_offset))
    selend = min(cutend, math.ceil(cutstart + (cutend - cutstart) * end_offset))
    # hid is (L, H)
    return hid[selstart:selend, :]

def separate_and_sample_data(data_array, tag_array, sample_size, tags=None):
    # Ensure data_array and tag_array are numpy arrays
    data_array = np.array(data_array)
    tag_array = np.array(tag_array)
    if tags is None: 
        # in this way we can provide tags externally and only select the data with those tags
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
def get_toplot(hiddens, sepframes1, sepframes2, phi_types, stop_names, offsets=(0, 1), contrast_in="asp", merge=True, hidden_dim=8, lookat="stop", include_map=None, aux_on=None): 
    # collect the start and end frames for each phoneme
    cutstarts = []
    cutends = []
    if lookat == "stop":
        for sepframe1, sepframe2, phi_type in zip(sepframes1, sepframes2, phi_types):
            # if phi_type == 'ST':
            #     cutstarts.append(sepframe1)
            # else:
            #     cutstarts.append(0)
            # we deleted the above code because now we have SIL. 
            cutstarts.append(sepframe1)
            cutends.append(sepframe2)
    elif lookat == "vowel": 
        for sepframe1, sepframe2, phi_type in zip(sepframes1, sepframes2, phi_types):
            # This is to get the vowel part
            cutstarts.append(sepframe2)
            cutends.append(None)
    elif lookat == "pre": 
        for sepframe1, sepframe2, phi_type in zip(sepframes1, sepframes2, phi_types):
            cutstarts.append(0)
            cutends.append(sepframe1)
    else: 
        raise ValueError("Lookat must be one of 'stop' or 'vowel'")
    
    if contrast_in == "asp": 
        tags_list = phi_types
    elif contrast_in == "stop":
        tags_list = stop_names
    elif contrast_in == "vowel":
        tags_list = stop_names  # should pass vowel_names to stop_names
    else:
        raise ValueError("Contrast_in must be one of 'asp' or 'stop'")
    
    if aux_on == "asp": 
        # aux_on is the auxiliary information that we want to for deciding the cut ranges that may not depend on the tag
        aux_on = phi_types
    elif aux_on == "stop": 
        aux_on = stop_names
    elif aux_on == "vowel":
        aux_on = stop_names
    else: 
        # meaning that we are not using any auxiliary information
        aux_on = tags_list
    
    hid_sel = np.empty((0, hidden_dim))
    tag_sel = []
    for (item, start, end, tag, auxtag) in zip(hiddens, cutstarts, cutends, tags_list, aux_on): 
        if include_map is not None and tag not in include_map.keys(): 
            continue

        if isinstance(offsets, tuple):
            offsetstart, offsetend = offsets
        elif isinstance(offsets, dict):
            offsetstart, offsetend = offsets[auxtag]
            # we let the model to report error if the tag is not in the dictionary
        else: 
            raise ValueError("Offsets must be either a tuple or a dictionary")
        
        hid = cutHid(item, start, end, offsetstart, offsetend)
        if merge:
            hid = np.mean(hid, axis=0, keepdims=True)
            hid_sel = np.concatenate((hid_sel, hid), axis=0)
            tag_sel += [include_map[tag] if include_map is not None else tag]
        else: 
            hidlen = hid.shape[0]
            hid_sel = np.concatenate((hid_sel, hid), axis=0)
            tag_sel += [include_map[tag] if include_map is not None else tag] * hidlen
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
    plt.ylabel('Silhouette Score (40%~60%)')
    plt.title('Silhouette Score Across Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_many(arrs, labels, save_path, plot_label_dict={"xlabel": "Epoch", "ylabel": "Value", "title": "Value Across Epochs"}): 
    n_steps = arrs[0].shape[1]
    mean_trajs = []
    lower_bounds = []
    upper_bounds = []
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for arr in arrs: 
        assert arr.shape[1] == n_steps
        mean_traj = np.mean(arr, axis=0)
        sem_arr = sem(arr, axis=0)
        ci_95 = 1.96 * sem_arr
        upper_bound = mean_traj + ci_95
        lower_bound = mean_traj - ci_95
        mean_trajs.append(mean_traj)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    # Plotting
    plt.figure(figsize=(12, 8))
    for idx, (mean_traj, lower_bound, upper_bound, label) in enumerate(zip(mean_trajs, lower_bounds, upper_bounds, labels)): 
        plt.plot(mean_traj, label=label, color=colors[idx])
        plt.fill_between(range(n_steps), lower_bound, upper_bound, color=colors[idx], alpha=0.2)

    plt.xlabel(plot_label_dict["xlabel"])
    plt.ylabel(plot_label_dict["ylabel"])
    plt.title(plot_label_dict["title"])
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--timestamp', '-ts', type=str, default="0000000000", help="Timestamp for project, better be generated by bash")
    parser.add_argument('--gpu', '-gpu', type=int, default=0, help="Choose the GPU to work on")
    parser.add_argument('--model','-m',type=str, default = "ae",help="Model type: ae or vqvae")
    parser.add_argument('--condition','-cd',type=str, default="b", help='Condition: b (balanced), u (unbalanced), nt (no-T)')
    parser.add_argument('--zlevel','-zl',type=str, default="hidrep", help='hidrep / attnout')
    parser.add_argument('--testname','-tn',type=str, default="abx-vowelstop", help='')
    args = parser.parse_args()

    # set device number
    torch.cuda.set_device(args.gpu)
    test_name = args.testname

    ts = args.timestamp # this timestamp does not contain run number
    train_name = "C_0Ta"
    res_save_dir = os.path.join(model_save_, f"eval-{train_name}-{ts}")
    model_type = args.model
    model_condition = args.condition
    zlevel = args.zlevel
    model_condition_dir = os.path.join(res_save_dir, model_type, model_condition)
    print(model_condition_dir)
    assert PU.path_exist(model_condition_dir)
    this_save_dir = os.path.join(model_condition_dir, "integrated_results")
    mk(this_save_dir)

    the_good_saving_dir = os.path.join(res_save_dir, test_name)
    mk(the_good_saving_dir)
    the_good_saving_dir_model_condition = os.path.join(res_save_dir, test_name, model_condition, model_type)
    mk(the_good_saving_dir_model_condition)

    hidden_dim = int(model_type.split("-")[0].replace("recon", "")) # get hidden dimension from model_type
    learned_runs = [1, 2, 3, 4, 5]
    string_learned_runs = [str(num) for num in learned_runs]
    strseq_learned_runs = "".join(string_learned_runs)

    stop_list_epochs = [] # list for each epoch of lists of sse for each run
    asp_list_epochs = []

    if test_name == "abx-vowelstop": 
        for epoch in range(0, 100): 
            # 先循环epoch，再循环run
            stop_list_runs = []
            asp_list_runs = []
            print(f"Processing {model_type} in epoch {epoch}...")
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
                all_vowel_names = hidrep["vn"]

                # Silhouette Score
                cluster_groups = ["P", "T", "K"]
                hidr_cs, tags_cs = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_stop_names,
                                                offsets=(0, 0.3), 
                                                contrast_in="stop", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="vowel")
                color_translate = {item: idx for idx, item in enumerate(cluster_groups)}
                hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)
                for i in range(2): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15)
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    abx_err02 = sym_abx_error(hidrs[0], hidrs[2], distance=euclidean_distance)
                    abx_err12 = sym_abx_error(hidrs[1], hidrs[2], distance=euclidean_distance)
                    stop_list_runs.append(abx_err01)
                    stop_list_runs.append(abx_err02)
                    stop_list_runs.append(abx_err12)

                # aspiration contrast
                cluster_groups = ["T", "ST"]
                hidr_cs, tags_cs = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_stop_names,
                                                offsets=(0, 0.3), 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="vowel")
                color_translate = {item: idx for idx, item in enumerate(cluster_groups)}
                hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)
                for i in range(6):
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15) # should be 2 only
                    abx_err = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    asp_list_runs.append(abx_err)

            stop_list_epochs.append(stop_list_runs) # 把每一个epoch的结果汇总，因为最后我们要保存结果，跑起来挺费时间的
            asp_list_epochs.append(asp_list_runs)

        stop_list_epochs = np.array(stop_list_epochs)
        asp_list_epochs = np.array(asp_list_epochs)
        stop_list_epochs = stop_list_epochs.transpose(1, 0)
        asp_list_epochs = asp_list_epochs.transpose(1, 0)
        plot_silhouette(asp_list_epochs, stop_list_epochs, os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"))
        np.save(os.path.join(res_save_dir, test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), stop_list_epochs)
        np.save(os.path.join(res_save_dir, test_name, f"05-save-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), asp_list_epochs)
        print("Done.")
    
    elif test_name == "abx-vowel": 
        for epoch in range(0, 100): 
            # 先循环epoch，再循环run
            stop_list_runs = []
            asp_list_runs = []
            print(f"Processing {model_type} in epoch {epoch}...")
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
                all_vowel_names = hidrep["vn"]

                cluster_groups = ["AA", "UW", "IY"]
                hidr_cs, tags_cs = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0, 1), 
                                                contrast_in="stop", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="vowel")
                color_translate = {item: idx for idx, item in enumerate(cluster_groups)}
                hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)
                for i in range(2): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15, tags=cluster_groups)
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    abx_err02 = sym_abx_error(hidrs[0], hidrs[2], distance=euclidean_distance)
                    abx_err12 = sym_abx_error(hidrs[1], hidrs[2], distance=euclidean_distance)
                    asp_list_runs.append(abx_err01)
                    asp_list_runs.append(abx_err02)
                    asp_list_runs.append(abx_err12)

                cluster_groups = ["P", "T", "K"]
                hidr_cs, tags_cs = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_stop_names,
                                                offsets=(0, 1), 
                                                contrast_in="stop", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="vowel")
                color_translate = {item: idx for idx, item in enumerate(cluster_groups)}
                hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)
                for i in range(2): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15, tags=cluster_groups)
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    abx_err02 = sym_abx_error(hidrs[0], hidrs[2], distance=euclidean_distance)
                    abx_err12 = sym_abx_error(hidrs[1], hidrs[2], distance=euclidean_distance)
                    stop_list_runs.append(abx_err01)
                    stop_list_runs.append(abx_err02)
                    stop_list_runs.append(abx_err12)

            stop_list_epochs.append(stop_list_runs) # 把每一个epoch的结果汇总，因为最后我们要保存结果，跑起来挺费时间的
            asp_list_epochs.append(asp_list_runs)

        stop_list_epochs = np.array(stop_list_epochs)
        asp_list_epochs = np.array(asp_list_epochs)
        stop_list_epochs = stop_list_epochs.transpose(1, 0)
        asp_list_epochs = asp_list_epochs.transpose(1, 0)
        plot_silhouette(asp_list_epochs, stop_list_epochs, os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"))
        np.save(os.path.join(res_save_dir, test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), stop_list_epochs)
        np.save(os.path.join(res_save_dir, test_name, f"05-save-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), asp_list_epochs)
        print("Done.")

    elif test_name == "abx-vowelstop-0.1": 
        # This is to see whether the first 10% of vowel could be used to predict the stop
        for epoch in range(0, 100): 
            # 先循环epoch，再循环run
            stop_list_runs = []
            asp_list_runs = []
            print(f"Processing {model_type} in epoch {epoch}...")
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
                                                offsets=(0, 0.1), 
                                                contrast_in="stop", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="vowel")
                color_translate = {item: idx for idx, item in enumerate(cluster_groups)}
                hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)
                for i in range(2): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15)
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    abx_err02 = sym_abx_error(hidrs[0], hidrs[2], distance=euclidean_distance)
                    abx_err12 = sym_abx_error(hidrs[1], hidrs[2], distance=euclidean_distance)
                    stop_list_runs.append(abx_err01)
                    stop_list_runs.append(abx_err02)
                    stop_list_runs.append(abx_err12)

                # aspiration contrast
                cluster_groups = ["T", "ST"]
                hidr_cs, tags_cs = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_stop_names,
                                                offsets=(0, 0.1), 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="vowel")
                color_translate = {item: idx for idx, item in enumerate(cluster_groups)}
                hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)
                for i in range(6):
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15) # should be 2 only
                    abx_err = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    asp_list_runs.append(abx_err)

            stop_list_epochs.append(stop_list_runs) # 把每一个epoch的结果汇总，因为最后我们要保存结果，跑起来挺费时间的
            asp_list_epochs.append(asp_list_runs)

        stop_list_epochs = np.array(stop_list_epochs)
        asp_list_epochs = np.array(asp_list_epochs)
        stop_list_epochs = stop_list_epochs.transpose(1, 0)
        asp_list_epochs = asp_list_epochs.transpose(1, 0)
        plot_silhouette(asp_list_epochs, stop_list_epochs, os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"))
        np.save(os.path.join(res_save_dir, test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), stop_list_epochs)
        np.save(os.path.join(res_save_dir, test_name, f"05-save-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), asp_list_epochs)
        print("Done.")
    elif test_name == "abx-prestop": 
        # This is to see whether the first 10% of vowel could be used to predict the stop
        for epoch in range(0, 100): 
            # 先循环epoch，再循环run
            stop_list_runs = []
            asp_list_runs = []
            print(f"Processing {model_type} in epoch {epoch}...")
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
                                                offsets=(0.8, 1), 
                                                contrast_in="stop", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="pre")
                color_translate = {item: idx for idx, item in enumerate(cluster_groups)}
                hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)
                for i in range(2): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15)
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    abx_err02 = sym_abx_error(hidrs[0], hidrs[2], distance=euclidean_distance)
                    abx_err12 = sym_abx_error(hidrs[1], hidrs[2], distance=euclidean_distance)
                    stop_list_runs.append(abx_err01)
                    stop_list_runs.append(abx_err02)
                    stop_list_runs.append(abx_err12)

                # aspiration contrast
                cluster_groups = ["T", "ST"]
                hidr_cs, tags_cs = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_stop_names,
                                                offsets=(0.8, 1), 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="pre")
                color_translate = {item: idx for idx, item in enumerate(cluster_groups)}
                hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)
                for i in range(6):
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15) # should be 2 only
                    abx_err = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    asp_list_runs.append(abx_err)

            stop_list_epochs.append(stop_list_runs) # 把每一个epoch的结果汇总，因为最后我们要保存结果，跑起来挺费时间的
            asp_list_epochs.append(asp_list_runs)

        stop_list_epochs = np.array(stop_list_epochs)
        asp_list_epochs = np.array(asp_list_epochs)
        stop_list_epochs = stop_list_epochs.transpose(1, 0)
        asp_list_epochs = asp_list_epochs.transpose(1, 0)
        plot_silhouette(asp_list_epochs, stop_list_epochs, os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"))
        np.save(os.path.join(res_save_dir, test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), stop_list_epochs)
        np.save(os.path.join(res_save_dir, test_name, f"05-save-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), asp_list_epochs)
        print("Done.")

    elif test_name == "abx-pph": 
        # this one evaluates the copntrast between p, p(h) and (p)h. 
        for epoch in range(0, 100): 
            # 先循环epoch，再循环run
            stop_list_runs = []
            asp_list_runs = []
            print(f"Processing {model_type} in epoch {epoch}...")
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
                all_vowel_names = hidrep["vn"]

                # Select ST
                hidr_p, tags_p = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0, 0.4), 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                include_map={"ST": "P"})
                # Select T (plosive)
                hidr_pp, tags_pp = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0, 0.25), 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                include_map={"T": "PP"})
                # Select T (aspiration)
                hidr_h, tags_h = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0.75, 1), 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                include_map={"T": "H"})
                # combine them
                hidr_cs, tags_cs = np.concatenate((hidr_p, hidr_pp, hidr_h), axis=0), np.concatenate((tags_p, tags_pp, tags_h), axis=0)
                hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)

                # Now we put in aspiration the contrast between pp and h
                for i in range(6): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15, tags=["PP", "H"])
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    asp_list_runs.append(abx_err01)

                # Now we put in stop the contrast between p and pp
                for i in range(6): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15, tags=["P", "PP"])
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    stop_list_runs.append(abx_err01)

            stop_list_epochs.append(stop_list_runs) # 把每一个epoch的结果汇总，因为最后我们要保存结果，跑起来挺费时间的
            asp_list_epochs.append(asp_list_runs)

        stop_list_epochs = np.array(stop_list_epochs)
        asp_list_epochs = np.array(asp_list_epochs)
        stop_list_epochs = stop_list_epochs.transpose(1, 0)
        asp_list_epochs = asp_list_epochs.transpose(1, 0)
        plot_silhouette(asp_list_epochs, stop_list_epochs, os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"))
        np.save(os.path.join(res_save_dir, test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), stop_list_epochs)
        np.save(os.path.join(res_save_dir, test_name, f"05-save-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), asp_list_epochs)
        print("Done.")
    elif test_name == "abx-pph-sm": 
        # this one evaluates the copntrast between p, p(h) and (p)h. 
        # avoid information overlap even more and use the middle part
        for epoch in range(0, 100): 
            # 先循环epoch，再循环run
            stop_list_runs = []
            asp_list_runs = []
            print(f"Processing {model_type} in epoch {epoch}...")
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
                all_vowel_names = hidrep["vn"]

                # Select ST
                hidr_p, tags_p = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0.2, 0.4), 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                include_map={"ST": "P"})
                # Select T (plosive)
                hidr_pp, tags_pp = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0.15, 0.3), 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                include_map={"T": "PP"})
                # Select T (aspiration)
                hidr_h, tags_h = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0.7, 0.85), 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                include_map={"T": "H"})
                # combine them
                hidr_cs, tags_cs = np.concatenate((hidr_p, hidr_pp, hidr_h), axis=0), np.concatenate((tags_p, tags_pp, tags_h), axis=0)
                hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)

                # Now we put in aspiration the contrast between pp and h
                for i in range(6): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15, tags=["PP", "H"])
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    asp_list_runs.append(abx_err01)

                # Now we put in stop the contrast between p and pp
                for i in range(6): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15, tags=["P", "PP"])
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    stop_list_runs.append(abx_err01)

            stop_list_epochs.append(stop_list_runs) # 把每一个epoch的结果汇总，因为最后我们要保存结果，跑起来挺费时间的
            asp_list_epochs.append(asp_list_runs)

        stop_list_epochs = np.array(stop_list_epochs)
        asp_list_epochs = np.array(asp_list_epochs)
        stop_list_epochs = stop_list_epochs.transpose(1, 0)
        asp_list_epochs = asp_list_epochs.transpose(1, 0)
        plot_silhouette(asp_list_epochs, stop_list_epochs, os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"))
        np.save(os.path.join(res_save_dir, test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), stop_list_epochs)
        np.save(os.path.join(res_save_dir, test_name, f"05-save-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), asp_list_epochs)
        print("Done.")
    elif test_name == "abx-pph-se": 
        # this one evaluates the copntrast between p, p(h) and (p)h. 
        # avoid information overlap even more and use the middle part
        for epoch in range(0, 100): 
            # 先循环epoch，再循环run
            stop_list_runs = []
            asp_list_runs = []
            print(f"Processing {model_type} in epoch {epoch}...")
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
                all_vowel_names = hidrep["vn"]

                # Select ST
                hidr_p, tags_p = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0, 0.2), 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                include_map={"ST": "P"})
                # Select T (plosive)tags_cs
                hidr_pp, tags_pp = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0, 0.15), 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                include_map={"T": "PP"})
                # Select T (aspiration)
                hidr_h, tags_h = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0.85, 1), 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                include_map={"T": "H"})
                # combine them
                hidr_cs, tags_cs = np.concatenate((hidr_p, hidr_pp, hidr_h), axis=0), np.concatenate((tags_p, tags_pp, tags_h), axis=0)
                hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)

                # Now we put in aspiration the contrast between pp and h
                for i in range(6): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15, tags=["PP", "H"])
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    asp_list_runs.append(abx_err01)

                # Now we put in stop the contrast between p and pp
                for i in range(6): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15, tags=["P", "PP"])
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    stop_list_runs.append(abx_err01)

            stop_list_epochs.append(stop_list_runs) # 把每一个epoch的结果汇总，因为最后我们要保存结果，跑起来挺费时间的
            asp_list_epochs.append(asp_list_runs)

        stop_list_epochs = np.array(stop_list_epochs)
        asp_list_epochs = np.array(asp_list_epochs)
        stop_list_epochs = stop_list_epochs.transpose(1, 0)
        asp_list_epochs = asp_list_epochs.transpose(1, 0)
        plot_silhouette(asp_list_epochs, stop_list_epochs, os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"))
        np.save(os.path.join(res_save_dir, test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), stop_list_epochs)
        np.save(os.path.join(res_save_dir, test_name, f"05-save-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), asp_list_epochs)
        print("Done.")

    elif test_name == "abx-pppptk" or test_name == "abx-pppptk-m": 
        # this one evaluates the copntrast between p, p(h) and (p)h. 
        # avoid information overlap even more and use the middle part
        for epoch in range(0, 100): 
            # 先循环epoch，再循环run
            stop_list_runs = []
            asp_list_runs = []
            print(f"Processing {model_type} in epoch {epoch}...")
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
                all_vowel_names = hidrep["vn"]

                if test_name == "abx-pppptk": 
                    offsets = {"ST": (0, 0.2), "T": (0, 0.15)}
                elif test_name == "abx-pppptk-m": 
                    offsets = {"ST": (0.2, 0.4), "T": (0.15, 0.3)}
                else: 
                    raise ValueError("Test_name must be one of 'abx-pppptk' or 'abx-pppptk-m'")

                # Select PPP (aspiration contrast)
                hidr_ppp, tags_ppp = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_stop_names,
                                                offsets=offsets, 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop")
                hidr_ppp, tags_ppp = postproc_standardize(hidr_ppp, tags_ppp, outlier_ratio=0.5)
                # Select PTK
                hidr_ptk, tags_ptk = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_stop_names,
                                                offsets=offsets, 
                                                contrast_in="stop", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                aux_on="asp")
                hidr_ptk, tags_ptk = postproc_standardize(hidr_ptk, tags_ptk, outlier_ratio=0.5)

                # Now we put in aspiration the contrast between p and pp (aspiration contrast, but only concerning burst)
                for i in range(6): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_ppp, tag_array=tags_ppp, sample_size=15, tags=["ST", "T"])
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    asp_list_runs.append(abx_err01)

                # Now we put in stop the contrast between p t and k (from burst only, same as aspiration condition)
                for i in range(2): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_ptk, tag_array=tags_ptk, sample_size=15)
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    abx_err02 = sym_abx_error(hidrs[0], hidrs[2], distance=euclidean_distance)
                    abx_err12 = sym_abx_error(hidrs[1], hidrs[2], distance=euclidean_distance)
                    stop_list_runs.append(abx_err01)
                    stop_list_runs.append(abx_err02)
                    stop_list_runs.append(abx_err12)

            stop_list_epochs.append(stop_list_runs) # 把每一个epoch的结果汇总，因为最后我们要保存结果，跑起来挺费时间的
            asp_list_epochs.append(asp_list_runs)

        stop_list_epochs = np.array(stop_list_epochs)
        asp_list_epochs = np.array(asp_list_epochs)
        stop_list_epochs = stop_list_epochs.transpose(1, 0)
        asp_list_epochs = asp_list_epochs.transpose(1, 0)
        plot_silhouette(asp_list_epochs, stop_list_epochs, os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"))
        np.save(os.path.join(res_save_dir, test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), stop_list_epochs)
        np.save(os.path.join(res_save_dir, test_name, f"05-save-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), asp_list_epochs)
        print("Done.")
    elif test_name == "abx-vowel-portion": 
        # this one evaluates the contrast between different portions of the vowel
        for epoch in range(0, 100): 
            # 先循环epoch，再循环run
            stop_list_runs = []
            asp_list_runs = []
            print(f"Processing {model_type} in epoch {epoch}...")
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
                all_vowel_names = hidrep["vn"]

                # if test_name == "abx-pppptk": 
                #     offsets = {"ST": (0, 0.2), "T": (0, 0.15)}
                # elif test_name == "abx-pppptk-m": 
                #     offsets = {"ST": (0.2, 0.4), "T": (0.15, 0.3)}
                # else: 
                #     raise ValueError("Test_name must be one of 'abx-pppptk' or 'abx-pppptk-m'")

                # Select portion 1 of vowels
                hidr_p_1, tags_p_1 = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0.1, 0.3), 
                                                contrast_in="vowel", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="vowel", 
                                                include_map={"AA": "1", "IY": "1", "UW": "1"})
                # Select portion 2 of vowels
                hidr_p_2, tags_p_2 = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0.4, 0.6), 
                                                contrast_in="vowel", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="vowel", 
                                                include_map={"AA": "2", "IY": "2", "UW": "2"})
                # Select portion 3 of vowels
                hidr_p_3, tags_p_3 = get_toplot(hiddens=all_zq,
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0.7, 0.9),
                                                contrast_in="vowel",
                                                merge=True,
                                                hidden_dim=hidden_dim,
                                                lookat="vowel",
                                                include_map={"AA": "3", "IY": "3", "UW": "3"})
                # concatenate the vowel portions
                hidr_p, tags_p = np.concatenate((hidr_p_1, hidr_p_2, hidr_p_3), axis=0), np.concatenate((tags_p_1, tags_p_2, tags_p_3), axis=0)
                hidr_p, tags_p = postproc_standardize(hidr_p, tags_p, outlier_ratio=0.5)


                # Select vowel type contrast
                hidr_v, tags_v = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0.3, 0.7), 
                                                contrast_in="vowel", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="vowel", 
                                                include_map={"AA": "AA", "IY": "IY", "UW": "UW"})
                hidr_v, tags_v = postproc_standardize(hidr_v, tags_v, outlier_ratio=0.5)

                # ASP: VOWEL PORTION CONTRAST
                for i in range(2): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_p, tag_array=tags_p, sample_size=15)
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    abx_err02 = sym_abx_error(hidrs[0], hidrs[2], distance=euclidean_distance)
                    abx_err12 = sym_abx_error(hidrs[1], hidrs[2], distance=euclidean_distance)
                    asp_list_runs.append(abx_err01)
                    asp_list_runs.append(abx_err02)
                    asp_list_runs.append(abx_err12)

                # STOP: VOWEL TYPE CONTRAST
                for i in range(2): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_v, tag_array=tags_v, sample_size=15)
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    abx_err02 = sym_abx_error(hidrs[0], hidrs[2], distance=euclidean_distance)
                    abx_err12 = sym_abx_error(hidrs[1], hidrs[2], distance=euclidean_distance)
                    stop_list_runs.append(abx_err01)
                    stop_list_runs.append(abx_err02)
                    stop_list_runs.append(abx_err12)

            stop_list_epochs.append(stop_list_runs) # 把每一个epoch的结果汇总，因为最后我们要保存结果，跑起来挺费时间的
            asp_list_epochs.append(asp_list_runs)

        stop_list_epochs = np.array(stop_list_epochs)
        asp_list_epochs = np.array(asp_list_epochs)
        stop_list_epochs = stop_list_epochs.transpose(1, 0)
        asp_list_epochs = asp_list_epochs.transpose(1, 0)
        plot_many([asp_list_epochs, stop_list_epochs], ["PORTION", "TYPE"], 
                  os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"), 
                  {"xlabel": "Epochs", "ylabel": "ABX Error Rate", "title": f"ABX Error Rate for {model_type} in {model_condition} at {zlevel}"})
        np.save(os.path.join(res_save_dir, test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), stop_list_epochs)
        np.save(os.path.join(res_save_dir, test_name, f"05-save-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), asp_list_epochs)
        print("Done.")
    elif test_name == "abx-pphb" or test_name == "abx-pphb-smallmiddle": 
        # this one evaluates the copntrast between p, p(h) and (p)h. 
        for epoch in range(0, 100): 
            # 先循环epoch，再循环run
            stop_list_runs = []
            asp_list_runs = []
            print(f"Processing {model_type} in epoch {epoch}...")
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
                elif zlevel == "ori": 
                    all_zq = hidrep["ori"]
                else: 
                    raise ValueError("zlevel must be one of 'hidrep' or 'attnout'")
                
                all_sepframes1 = hidrep["sep-frame1"]
                all_sepframes2 = hidrep["sep-frame2"]
                all_phi_type = hidrep["phi-type"]
                all_stop_names = hidrep["sn"]
                all_vowel_names = hidrep["vn"]

                if test_name == "abx-pphb": 
                    offsets = {"B": (0, 0.4), "PP": (0, 0.25), "H": (0.75, 1)}
                elif test_name == "abx-pphb-smallmiddle": 
                    offsets = {"B": (0.16, 0.32), "PP": (0.1, 0.2), "H": (0.8, 0.9)}
                else: 
                    raise ValueError("Test_name must be one of 'abx-pphb' or 'abx-pphb-smallmiddle'")

                # Select ST
                hidr_p, tags_p = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=offsets["B"], 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                include_map={"D": "B"})
                # Select T (plosive)
                hidr_pp, tags_pp = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=offsets["PP"], 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                include_map={"T": "PP"})
                # Select T (aspiration)
                hidr_h, tags_h = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=offsets["H"], 
                                                contrast_in="asp", 
                                                merge=True, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                include_map={"T": "H"})
                # combine them
                hidr_cs, tags_cs = np.concatenate((hidr_p, hidr_pp, hidr_h), axis=0), np.concatenate((tags_p, tags_pp, tags_h), axis=0)
                hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)

                # Now we put in aspiration the contrast between pp and h
                for i in range(6): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15, tags=["PP", "H"])
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    asp_list_runs.append(abx_err01)

                # Now we put in stop the contrast between p and pp
                for i in range(6): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15, tags=["B", "PP"])
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=euclidean_distance)
                    stop_list_runs.append(abx_err01)

            stop_list_epochs.append(stop_list_runs) # 把每一个epoch的结果汇总，因为最后我们要保存结果，跑起来挺费时间的
            asp_list_epochs.append(asp_list_runs)

        stop_list_epochs = np.array(stop_list_epochs)
        asp_list_epochs = np.array(asp_list_epochs)
        stop_list_epochs = stop_list_epochs.transpose(1, 0)
        asp_list_epochs = asp_list_epochs.transpose(1, 0)
        plot_silhouette(asp_list_epochs, stop_list_epochs, os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"))
        plot_many([asp_list_epochs, stop_list_epochs], ["PPH", "PB"], 
                  os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"), 
                  {"xlabel": "Epochs", "ylabel": "ABX Error Rate", "title": f"ABX Error Rate for {model_type} in {model_condition} at {zlevel}"})
        np.save(os.path.join(res_save_dir, test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), stop_list_epochs)
        np.save(os.path.join(res_save_dir, test_name, f"05-save-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), asp_list_epochs)
        print("Done.")