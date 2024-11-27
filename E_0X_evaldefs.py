"""
Here we define the evaluation functions for the E_0X models.
"""

from C_0X_defs import *
from C_0Y_evaldefs import *
import plotly.graph_objs as go
import plotly.express as px
from colorama import Fore, Back, Style

model_configs = {
    "hiddim": 96, 
    "ori_select_dim": 64,   # this is for choosing the dimension to calculate ori, but because ori the same across all dimensions, we just use 64
}

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
    if lookat == "second":
        for sepframe1, sepframe2, phi_type in zip(sepframes1, sepframes2, phi_types):
            cutstarts.append(sepframe1)
            cutends.append(sepframe2)
    elif lookat == "third": 
        for sepframe1, sepframe2, phi_type in zip(sepframes1, sepframes2, phi_types):
            # This is to get the vowel part
            cutstarts.append(sepframe2)
            cutends.append(None)
    elif lookat == "first": 
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
    elif contrast_in == "pre": 
        tags_list = stop_names
    elif contrast_in == "data": 
        tags_list = stop_names
    else:
        raise ValueError("Contrast_in must be one of 'asp' or 'stop'")
    
    if aux_on == "asp": 
        # aux_on is the auxiliary information that we want to for deciding the cut ranges that may not depend on the tag
        aux_on = phi_types
    elif aux_on == "data": 
        aux_on = stop_names
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
    colors = [
    "#da1e28", "#f1c21b", "#ff832b", "#198038",
    "#edf5ff", "#f6f2ff", "#d9fbfb", 
    "#a6c8ff", "#d4bbff", "#3ddbd9", 
    "#4589ff", "#a56eff", "#009d9a",
    "#0043ce", "#6929c4", "#005d5d",
    "#001d6c", "#31135e", "#022b30", "black"
    ]

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

def plot_many_plotly(arrs, labels, save_path, plot_label_dict={"xlabel": "Epoch", "ylabel": "Value", "title": "Value Across Epochs"}, y_range=None, cloud=True):
    n_steps = arrs[0].shape[1]
    mean_trajs = []
    lower_bounds = []
    upper_bounds = []
    colors = px.colors.qualitative.Alphabet

    for arr in arrs:
        assert arr.shape[1] == n_steps
        mean_traj = np.mean(arr, axis=0)
        sem_arr = stats.sem(arr, axis=0)
        ci_95 = 1.96 * sem_arr
        upper_bound = mean_traj + ci_95
        lower_bound = mean_traj - ci_95
        mean_trajs.append(mean_traj)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    # Create Plotly traces
    fig = go.Figure()
    for idx, (mean_traj, lower_bound, upper_bound, label) in enumerate(zip(mean_trajs, lower_bounds, upper_bounds, labels)):
        # Add confidence interval shading if cloud is True
        if cloud:
            fig.add_trace(go.Scatter(
                x=list(range(n_steps)) + list(range(n_steps))[::-1],
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself',
                fillcolor=colors[idx],
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                opacity=0.2,
                legendgroup=label,
            ))
        # Add mean line
        fig.add_trace(go.Scatter(
            x=list(range(n_steps)),
            y=mean_traj,
            mode='lines',
            name=label,
            line=dict(color=colors[idx]),
            legendgroup=label,
        ))

    # Update layout
    fig.update_layout(
        title=plot_label_dict["title"],
        xaxis_title=plot_label_dict["xlabel"],
        yaxis_title=plot_label_dict["ylabel"],
        yaxis=dict(range=y_range if y_range is not None else [None, None]),
        legend_title="Trajectories",
        template="plotly_white"
    )
    
    # Save to HTML file
    fig.write_html(save_path)

def plot_many_plotly_errbar(arrs, labels, save_path, plot_label_dict={"xlabel": "Epoch", "ylabel": "Value", "title": "Value Across Epochs"}, x_list=[4, 8, 16, 32, 48, 64], y_range=None, cloud=True):
    n_steps = len(x_list)
    mean_trajs = []
    lower_bounds = []
    upper_bounds = []
    ci_95s = []
    colors = px.colors.qualitative.Alphabet

    for arr in arrs:
        assert arr.shape[1] == n_steps
        mean_traj = np.mean(arr, axis=0)
        sem_arr = stats.sem(arr, axis=0)
        ci_95 = 1.96 * sem_arr
        upper_bound = mean_traj + ci_95
        lower_bound = mean_traj - ci_95
        mean_trajs.append(mean_traj)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
        ci_95s.append(ci_95)

    # Create Plotly traces
    fig = go.Figure()
    for idx, (mean_traj, lower_bound, upper_bound, ci_95, label) in enumerate(zip(mean_trajs, lower_bounds, upper_bounds, ci_95s, labels)):
        # Add confidence interval error bar if cloud is True
        if cloud:
            fig.add_trace(go.Scatter(
                x=x_list,
                y=mean_traj,
                mode='lines+markers',  # Markers for error bars
                name=label,
                marker=dict(color=colors[idx], size=8),
                error_y=dict(
                    type='data',
                    array=ci_95,  # Error bars using CI_95
                    visible=True
                )
            ))
        else: 
            fig.add_trace(go.Scatter(
                x=x_list,
                y=mean_traj,
                mode='lines+markers',
                name=label,
                line=dict(color=colors[idx]),
            ))

    # Update layout
    fig.update_layout(
        title=plot_label_dict["title"],
        xaxis_title=plot_label_dict["xlabel"],
        yaxis_title=plot_label_dict["ylabel"],
        yaxis=dict(range=y_range if y_range is not None else [None, None]),
        legend_title="Representations",
        template="plotly_white"
    )
    
    # Save to HTML file
    fig.write_html(save_path)



def get_representation(data_collection, representation_select, hidden_dim_required): 
    hidden_dim_use = hidden_dim_required
    other_hid_outs = data_collection["other-hid-outs"]
    if representation_select == "hidrep": 
        all_representations = data_collection["ze"]
    elif representation_select == "attnout": 
        all_representations = data_collection["zq"]
    elif representation_select == "ori": 
        if hidden_dim_required != model_configs["ori_select_dim"]: 
            # raise Exception("Warning: hidden_dim is not 64, but we are using the original representation! ")
            print(Fore.RED + "Warning: hidden_dim is not 64, but using the original representation! ")
            raise SystemExit()
        all_representations = data_collection["ori"]
        hidden_dim_use = model_configs["hiddim"]
    elif representation_select in other_hid_outs.keys(): 
        all_representations = other_hid_outs[representation_select]
    else: 
        raise ValueError("zlevel must be one of 'hidrep' or 'attnout'")
    return all_representations, hidden_dim_use