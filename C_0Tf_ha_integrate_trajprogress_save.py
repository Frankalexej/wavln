import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# from C_0B_eval import *
from C_0X_defs import *
from scipy.stats import sem, ttest_ind

def read_result_at(res_save_dir, epoch): 
    all_handler = DictResHandler(whole_res_dir=res_save_dir, 
                                 file_prefix=f"all-{epoch}")

    all_handler.read()

    return all_handler.res

def calculate_means_and_sems(values):
    """Calculates means and standard errors of the means (SEMs) for input values."""
    return np.mean(values), sem(values)

def plot_attention_epoch_trajectory(all_phi_type, all_attn, all_sepframes1, all_sepframes2, save_path, conditionlist=["ST", "T"]): 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    legend_namess = [['S-to-P', 'P-to-S', 'P-to-V', 'V-to-P'], ['#-to-P', 'P-to-#', 'P-to-V', 'V-to-P']]
    colors = ['b', 'g', 'red', 'orange']
    n_steps = 100
    segment_length = int(n_steps * 0.1)  # Calculate 20% segment length
    badcounts = {selector : 0 for selector in conditionlist}
    totalcounts = {selector : 0 for selector in conditionlist}

    non_mean_attnres = {}
    non_mean_attnres_merge = {}
    for (selector, ax, legend_names) in zip(conditionlist, [ax1, ax2], legend_namess):
        # 这个只是处理ST和T，而非循环
        meanslist = []
        upperlist = []
        lowerlist = []
        non_mean_attnres[selector] = {}
        non_mean_attnres[selector]["s2t"] = []
        non_mean_attnres[selector]["t2s"] = []
        non_mean_attnres[selector]["t2a"] = []
        non_mean_attnres[selector]["a2t"] = []
        for epoch in range(len(all_attn)): 
            # 循环每个epoch
            phi_type_epoch = all_phi_type[epoch]
            attn_epoch = all_attn[epoch]
            sepframes1_epoch = all_sepframes1[epoch]
            sepframes2_epoch = all_sepframes2[epoch]
            selected_tuples = [(sf1, sf2, attn) for pt, sf1, sf2, attn in zip(phi_type_epoch,  
                                                            sepframes1_epoch, 
                                                            sepframes2_epoch, 
                                                            attn_epoch) if pt == selector]
            selected_sf1s_epoch, selected_sf2s_epoch, selected_attns_epoch = zip(*selected_tuples)
            s_to_t_traj = []
            t_to_s_traj = []
            t_to_a_traj = []
            a_to_t_traj = []

            totalcounts[selector] += len(selected_attns_epoch)
        
            for i in range(len(selected_attns_epoch)): 
                # 循环每个run
                this_attn = selected_attns_epoch[i]
                # this_sep_frame0 = selected_sf0s_epoch[i]
                this_sep_frame1 = selected_sf1s_epoch[i]
                this_sep_frame2 = selected_sf2s_epoch[i]

                if selector == "ST": 
                    blocks = extract_attention_blocks_ST(this_attn, this_sep_frame1, this_sep_frame2)
                elif selector in ["T", "D", "TT"]: 
                    blocks = extract_attention_blocks_ST(this_attn, this_sep_frame1, this_sep_frame2)
                else: 
                    raise ValueError("selector must be ST or T")

                # s_to_t_interp = interpolate_traj(blocks['s_to_t'], n_steps)
                # t_to_s_interp = interpolate_traj(blocks['t_to_s'], n_steps)
                # t_to_a_interp = interpolate_traj(blocks['t_to_a'], n_steps)
                # a_to_t_interp = interpolate_traj(blocks['a_to_t'], n_steps)
                s_to_t_interp = blocks['s_to_t']
                t_to_s_interp = blocks['t_to_s']
                t_to_a_interp = blocks['t_to_a']
                a_to_t_interp = blocks['a_to_t']

                if np.any(np.isnan(s_to_t_interp)) or np.any(np.isnan(t_to_s_interp)) or np.any(np.isnan(t_to_a_interp)) or np.any(np.isnan(a_to_t_interp)):
                    badcounts[selector] += 1
                    # print(f"NAN at {epoch} in run {i} for {selector}")
                    continue
                s_to_t_traj.append(s_to_t_interp[-1])
                t_to_s_traj.append(t_to_s_interp[0])
                t_to_a_traj.append(t_to_a_interp[-1])
                a_to_t_traj.append(a_to_t_interp[0])

            # Convert list of arrays into 2D NumPy arrays for easier manipulation
            group1_array = np.array(s_to_t_traj)
            group2_array = np.array(t_to_s_traj)
            group3_array = np.array(t_to_a_traj)
            group4_array = np.array(a_to_t_traj)

            target_group1 = group1_array.flatten()
            target_group2 = group2_array.flatten()
            target_group3 = group3_array.flatten()
            target_group4 = group4_array.flatten()
            
            non_mean_attnres[selector]["s2t"].append(target_group1)
            non_mean_attnres[selector]["t2s"].append(target_group2)
            non_mean_attnres[selector]["t2a"].append(target_group3)
            non_mean_attnres[selector]["a2t"].append(target_group4)
        
        non_mean_attnres_merge[selector] = {}
        non_mean_attnres_merge[selector]["s2t"] = np.array(non_mean_attnres[selector]["s2t"])
        non_mean_attnres_merge[selector]["t2s"] = np.array(non_mean_attnres[selector]["t2s"])
        non_mean_attnres_merge[selector]["t2a"] = np.array(non_mean_attnres[selector]["t2a"])
        non_mean_attnres_merge[selector]["a2t"] = np.array(non_mean_attnres[selector]["a2t"])

    print(f"badcounts: {badcounts}")
    print(f"totalcounts: {totalcounts}")
    with open(save_path, 'wb') as f:
        pickle.dump(non_mean_attnres_merge, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--timestamp', '-ts', type=str, default="0000000000", help="Timestamp for project, better be generated by bash")
    parser.add_argument('--gpu', '-gpu', type=int, default=0, help="Choose the GPU to work on")
    parser.add_argument('--model','-m',type=str, default = "ae",help="Model type: ae or vqvae")
    parser.add_argument('--condition','-cd',type=str, default="b", help='Condition: b (balanced), u (unbalanced), nt (no-T)')
    args = parser.parse_args()

    ts = args.timestamp
    model_type = args.model
    model_condition = args.condition
    train_name = "C_0Tf"
    root_ = "/mnt/storage/compling/wavln/"
    model_save_ = root_ + "model_save/"
    res_save_dir = os.path.join(model_save_, f"eval-{train_name}-{ts}")

    sil_dict = {}
    model_condition_dir = os.path.join(res_save_dir, model_type, model_condition)
    assert PU.path_exist(model_condition_dir)
    this_save_dir = os.path.join(model_condition_dir, "integrated_results")
    mk(this_save_dir)

    every_attns = []
    # every_sepframes0 = []
    every_sepframes1 = []
    every_sepframes2 = []
    every_phi_types = []

    learned_runs = [1, 2, 3, 4, 5]
    string_learned_runs = [str(num) for num in learned_runs]
    strseq_learned_runs = "".join(string_learned_runs)

    for epoch in range(0, 101): 
        cat_attns = []
        # cat_sepframes0 = []
        cat_sepframes1 = []
        cat_sepframes2 = []
        cat_phi_types = []
        print(f"Processing {model_type} at {epoch}...")

        for run_number in learned_runs:
            this_model_condition_dir = os.path.join(model_condition_dir, f"{run_number}")
            allres = read_result_at(this_model_condition_dir, epoch)
            cat_phi_types += allres["phi-type"]
            cat_attns += allres["attn"]
            # cat_sepframes0 += allres["sep-frame0"]
            cat_sepframes1 += allres["sep-frame1"]
            cat_sepframes2 += allres["sep-frame2"]

        # plot_attention_comparison(cat_phi_types, cat_attns, cat_sepframes1, cat_sepframes2, os.path.join(this_save_dir, f"attncomp-at-{epoch}.png"))
        every_attns.append(cat_attns)
        # every_sepframes0.append(cat_sepframes0)
        every_sepframes1.append(cat_sepframes1)
        every_sepframes2.append(cat_sepframes2)
        every_phi_types.append(cat_phi_types)    
    plot_attention_epoch_trajectory(every_phi_types, every_attns, every_sepframes1, every_sepframes2, os.path.join(res_save_dir, f"attnepochtraj-at-all-{model_type}-{model_condition}-{strseq_learned_runs}.attnres"), 
                                    conditionlist=["T", "ST"])

    print("Done.")