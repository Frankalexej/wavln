"""
In this script, we will use vowels but stop consonant tags to evaluate the model. 
This is because stops are short and listeners usually use acoustic cues from the vowel 
to identify the stop consonant. We will check whether the model also did the same. 
"""

from cgi import test
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

def plot_many(arrs, labels, save_path, plot_label_dict={"xlabel": "Epoch", "ylabel": "Value", "title": "Value Across Epochs"}, y_range=None, cloud=True): 
    n_steps = arrs[0].shape[1]
    mean_trajs = []
    lower_bounds = []
    upper_bounds = []
    # colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 
    #           'black', 'yellow', 'magenta', 'lime', 'teal', 'lavender', 'tan', 'salmon', 'gold', 'indigo']
    colors = ['#1f77b4',  # blue
          '#d62728',  # red                
          '#ff7f0e',  # orange
          '#2ca02c',  # green
          '#9467bd',  # purple
          '#8c564b',  # brown
          '#e377c2',  # pink
          '#7f7f7f',  # gray
          '#bcbd22',  # yellow-green
          '#17becf',  # teal
          '#aec7e8',  # light blue
          '#ffbb78',  # light orange
          '#98df8a',  # light green
          '#ff9896',  # light red
          '#c5b0d5']  # light purple

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
        if cloud: 
            plt.fill_between(range(n_steps), lower_bound, upper_bound, color=colors[idx], alpha=0.2)

    if y_range is not None: 
        plt.ylim(y_range)
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
    train_name = "C_0Te"
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


    if test_name in ["abx-pph", "abx-pph-0903-1", "abx-pph-0903-2"]: 
        # 1 is euclidean, 2 is cosine distance
        # this one evaluates the copntrast between p, p(h) and (p)h. 
        for epoch in range(0, 101): 
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
                # select representation to work on
                other_hid_outs = hidrep["other-hid-outs"]
                if zlevel == "hidrep": 
                    all_zq = hidrep["ze"]
                elif zlevel == "attnout": 
                    all_zq = hidrep["zq"]
                elif zlevel == "ori": 
                    if hidden_dim != 64: 
                        raise Exception("Warning: hidden_dim is not 64, but we are using the original representation! ")
                    all_zq = hidrep["ori"]
                elif zlevel in other_hid_outs.keys(): 
                    all_zq = other_hid_outs[zlevel]
                else: 
                    raise ValueError("zlevel must be one of 'hidrep' or 'attnout'")
                
                all_sepframes1 = hidrep["sep-frame1"]
                all_sepframes2 = hidrep["sep-frame2"]
                all_phi_type = hidrep["phi-type"]
                all_stop_names = hidrep["sn"]
                all_vowel_names = hidrep["vn"]

                if test_name in ["abx-pph", "abx-pph-0903", "abx-pph-0903-1", "abx-pph-0903-2"]: 
                    # offsets = {"P": (0.3, 0.5), "PP": (0.15, 0.2), "H": (0.65, 0.7)}
                    # offsets = {"P": (0.4, 0.55), "PP": (0.15, 0.2), "H": (0.75, 0.8)}
                    offsets = {"P": (0.3, 0.45), "PP": (0.15, 0.2), "H": (0.85, 0.9)}
                else: 
                    raise ValueError("Test_name must be one of 'abx-pphb' or 'abx-pphb-smallmiddle'")
                
                merge_one_vector = False
                # Select ST
                hidr_p, tags_p = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=offsets["P"], 
                                                contrast_in="asp", 
                                                merge=merge_one_vector, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                include_map={"TT": "P"})
                # Select T (plosive)
                hidr_pp, tags_pp = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=offsets["PP"], 
                                                contrast_in="asp", 
                                                merge=merge_one_vector, 
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
                                                merge=merge_one_vector, 
                                                hidden_dim=hidden_dim, 
                                                lookat="stop", 
                                                include_map={"T": "H"})
                # combine them
                hidr_cs, tags_cs = np.concatenate((hidr_p, hidr_pp, hidr_h), axis=0), np.concatenate((tags_p, tags_pp, tags_h), axis=0)
                hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)


                if test_name in ["abx-pph-0903-2"]: 
                    distance_metrics = cosine_distance
                else: 
                    distance_metrics = euclidean_distance

                # Now we put in aspiration the contrast between pp and h
                for i in range(6): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15, tags=["PP", "H"])
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=distance_metrics)
                    asp_list_runs.append(abx_err01)

                # Now we put in stop the contrast between p and pp
                for i in range(6): 
                    hidrs, tagss = separate_and_sample_data(data_array=hidr_cs, tag_array=tags_cs, sample_size=15, tags=["P", "PP"])
                    abx_err01 = sym_abx_error(hidrs[0], hidrs[1], distance=distance_metrics)
                    stop_list_runs.append(abx_err01)

            stop_list_epochs.append(stop_list_runs) # 把每一个epoch的结果汇总，因为最后我们要保存结果，跑起来挺费时间的
            asp_list_epochs.append(asp_list_runs)

        stop_list_epochs = np.array(stop_list_epochs)
        asp_list_epochs = np.array(asp_list_epochs)
        stop_list_epochs = stop_list_epochs.transpose(1, 0)
        asp_list_epochs = asp_list_epochs.transpose(1, 0)
        # plot_silhouette(asp_list_epochs, stop_list_epochs, os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"))
        plot_many([asp_list_epochs, stop_list_epochs], ["PPH", "PPP"], 
                  os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"), 
                  {"xlabel": "Epochs", "ylabel": "ABX Error Rate", "title": f"ABX Error Rate for {model_type} in {model_condition} at {zlevel}"}, 
                  y_range=(0, 0.6))
        np.save(os.path.join(res_save_dir, test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), stop_list_epochs)
        np.save(os.path.join(res_save_dir, test_name, f"05-save-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), asp_list_epochs)
        print("Done.")
    elif test_name.split("-")[-1] == "catch": 
        uncatched_test_name = test_name.rsplit("-", 1)[0]
        stop_list_epochs = np.load(os.path.join(res_save_dir, uncatched_test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"))
        asp_list_epochs = np.load(os.path.join(res_save_dir, uncatched_test_name, f"05-save-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"))
        plot_many([asp_list_epochs, stop_list_epochs], ["PPH", "PPP"], 
                  os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"), 
                  {"xlabel": "Epochs", "ylabel": "ABX Error Rate", "title": f"ABX Error Rate for {model_type} in {model_condition} at {zlevel}"}, 
                  y_range=(0, 0.6))

        print("Done.")
    elif test_name.split("-")[0] in ["clusterARI"]: 
        # 1 is euclidean, 2 is cosine distance
        # this one evaluates the copntrast between p, p(h) and (p)h. 
        for epoch in range(0, 101): 
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
                # select representation to work on
                other_hid_outs = hidrep["other-hid-outs"]
                if zlevel == "hidrep": 
                    all_zq = hidrep["ze"]
                elif zlevel == "attnout": 
                    all_zq = hidrep["zq"]
                elif zlevel == "ori": 
                    if hidden_dim != 64: 
                        raise Exception("Warning: hidden_dim is not 64, but we are using the original representation! ")
                    all_zq = hidrep["ori"]
                elif zlevel in other_hid_outs.keys(): 
                    all_zq = other_hid_outs[zlevel]
                else: 
                    raise ValueError("zlevel must be one of 'hidrep' or 'attnout'")
                
                all_sepframes1 = hidrep["sep-frame1"]
                all_sepframes2 = hidrep["sep-frame2"]
                all_phi_type = hidrep["phi-type"]
                all_stop_names = hidrep["sn"]
                all_vowel_names = hidrep["vn"]

                test_name_lookat = test_name.split("-")[1]
                test_name_label = test_name.split("-")[2]
                
                merge_one_vector = False
                # Select Vowels and Vowel Tags
                hidr_p, tags_p = get_toplot(hiddens=all_zq, 
                                                sepframes1=all_sepframes1,
                                                sepframes2=all_sepframes2,
                                                phi_types=all_phi_type,
                                                stop_names=all_vowel_names,
                                                offsets=(0.4, 0.6), 
                                                contrast_in=test_name_label, 
                                                merge=merge_one_vector, 
                                                hidden_dim=hidden_dim, 
                                                lookat=test_name_lookat, 
                                                include_map={"AA": "AA", 
                                                             "UW": "UW",
                                                             "IY": "IY"})
                # combine them
                # hidr_cs, tags_cs = np.concatenate((hidr_p, hidr_pp, hidr_h), axis=0), np.concatenate((tags_p, tags_pp, tags_h), axis=0)
                hidr_cs, tags_cs = hidr_p, tags_p
                hidr_cs, tags_cs = postproc_standardize(hidr_cs, tags_cs, outlier_ratio=0.5)

                kmeans = KMeans(n_clusters=3, random_state=0)
                predicted_labels = kmeans.fit_predict(hidr_cs)
                ari = adjusted_rand_score(tags_cs, predicted_labels)
                asp_list_runs.append(ari)

            asp_list_epochs.append(asp_list_runs)

        asp_list_epochs = np.array(asp_list_epochs)
        asp_list_epochs = asp_list_epochs.transpose(1, 0)
        # plot_silhouette(asp_list_epochs, stop_list_epochs, os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"))
        plot_many([asp_list_epochs], ["ARI"], 
                  os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"), 
                  {"xlabel": "Epochs", "ylabel": "ABX Error Rate", "title": f"ABX Error Rate for {model_type} in {model_condition} at {zlevel}"}, 
                  y_range=(0, 1.0))
        # np.save(os.path.join(res_save_dir, test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), stop_list_epochs)
        np.save(os.path.join(res_save_dir, test_name, f"07-save-ari-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), asp_list_epochs)
        print("Done.")

    elif test_name.split("-")[0] in ["clusterARIAll"]: 
        # 1 is euclidean, 2 is cosine distance
        # this one evaluates the copntrast between p, p(h) and (p)h.
        # 'dec-lin1' 'enc-rnn1-f' 'enc-rnn1-b' 'dec-rnn1-f' 'enc-rnn2-f' 'enc-rnn2-b' 'dec-rnn2-f' 
        layered_res = {}
        for layer in ["hidrep", "attnout", "dec-lin1", "enc-lin1", 
                      "dec-rnn1-f", "enc-rnn1-f", "enc-rnn1-b",
                      "dec-rnn2-f", "enc-rnn2-f", "enc-rnn2-b"]: 
            print(f"Processing {model_type} in layer {layer}...")
            asp_list_epochs = []
            look_for_layer_path = test_name.split("-")[0][:-3] + "-" + test_name.split("-")[1] + "-" + test_name.split("-")[2]
            layer_path = os.path.join(res_save_dir, look_for_layer_path, 
                                      f"07-save-ari-{model_type}-{model_condition}-{strseq_learned_runs}-{layer}.npy")
            layer_res = np.load(layer_path)
            layered_res[layer] = layer_res
        # deal with ori
        ori_path = os.path.join(res_save_dir, test_name, f"07-save-ari-recon64-phi-{model_condition}-{strseq_learned_runs}-ori.npy")
        if os.path.exists(ori_path): 
            ori_res = np.load(ori_path)
            layered_res["ori"] = ori_res
        plot_many(list(layered_res.values()), list(layered_res.keys()), 
                  os.path.join(res_save_dir, test_name, f"03-stat-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.png"), 
                  {"xlabel": "Epochs", "ylabel": "ABX Error Rate", "title": f"ABX Error Rate for {model_type} in {model_condition} at {zlevel}"}, 
                  y_range=(0, 0.8), cloud=False)
        # np.save(os.path.join(res_save_dir, test_name, f"04-save-ptk-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), stop_list_epochs)
        # np.save(os.path.join(res_save_dir, test_name, f"05-save-asp-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.npy"), asp_list_epochs)
        # with open(os.path.join(res_save_dir, test_name, f"06-save-ari-{model_type}-{model_condition}-{strseq_learned_runs}-{zlevel}.pkl"), "wb") as f: 
        #     pickle.dump(layered_res, f)
        print("Done.")