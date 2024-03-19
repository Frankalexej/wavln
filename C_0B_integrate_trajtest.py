import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from C_0B_eval import *
from scipy.stats import sem, ttest_ind

def read_result_at(res_save_dir, epoch): 
    all_handler = DictResHandler(whole_res_dir=res_save_dir, 
                                 file_prefix=f"all-{epoch}")

    all_handler.read()

    return all_handler.res

def calculate_means_and_sems(values):
    """Calculates means and standard errors of the means (SEMs) for input values."""
    return np.mean(values), sem(values)

def plot_attention_comparison(all_phi_type, all_attn, all_sepframes1, all_sepframes2, save_path): 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    legend_names = ['S-to-P', 'P-to-S', 'P-to-V', 'V-to-P']
    colors = ['blue', 'green', 'red', 'orange']
    n_steps = 100
    segment_length = int(n_steps * 0.2)  # Calculate 20% segment length

    for (selector, ax) in zip(["ST", "T"], [ax1, ax2]):
        selected_tuples = [(sf1, sf2, attn) for pt, sf1, sf2, attn in zip(all_phi_type,  
                                                          all_sepframes1, 
                                                          all_sepframes2, 
                                                          all_attn) if pt == selector]
        selected_sf1s, selected_sf2s, selected_attns = zip(*selected_tuples)
        if selector == "ST":
            s_to_t_traj = []
            t_to_s_traj = []
            t_to_a_traj = []
            a_to_t_traj = []
            for i in range(len(selected_attns)): 
                this_attn = selected_attns[i]
                this_sep_frame1 = selected_sf1s[i]
                this_sep_frame2 = selected_sf2s[i]

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

            target_group1 = group1_array[:, -segment_length:].flatten()
            target_group2 = group2_array[:, :segment_length].flatten()
            target_group3 = group3_array[:, -segment_length:].flatten()
            target_group4 = group4_array[:, :segment_length].flatten()

            # Calculate the mean trajectory for each group
            means = np.array([np.mean(target_group1, axis=0), 
                            np.mean(target_group2, axis=0), 
                            np.mean(target_group3, axis=0), 
                            np.mean(target_group4, axis=0)])

            # Calculate the SEM for each step in both groups
            sems = np.array([sem(target_group1, axis=0),
                            sem(target_group2, axis=0),
                            sem(target_group3, axis=0),
                            sem(target_group4, axis=0)])
            
            # Perform statistical tests between the three specified pairs
            _, p_val_s2p_vs_p2s = ttest_ind(target_group1, target_group2)
            _, p_val_p2s_vs_p2v = ttest_ind(target_group2, target_group3)
            _, p_val_p2v_vs_v2p = ttest_ind(target_group3, target_group4)

            use_labels = legend_names
            use_colors = colors

            x = np.arange(len(use_labels))  # Label locations
            bars = ax.bar(x, means, yerr=1.96*np.array(sems), capsize=5, color=use_colors)

            # Mark significance directly on the bar graph
            significance_threshold_1 = 0.05
            significance_threshold_2 = 0.01
            significance_threshold_3 = 0.001
            signif_positions_pairs = [(0, 1), (1, 2), (2, 3)]  # Pairs of positions for each comparison
            p_values = [p_val_s2p_vs_p2s, p_val_p2s_vs_p2v, p_val_p2v_vs_v2p]

            for (pos1, pos2), p_val in zip(signif_positions_pairs, p_values):
                y_max = max(means[pos1] + 1.96*sems[pos1], means[pos2] + 1.96*sems[pos2])
                h = y_max * 0.05  # 5% above the max for drawing the line
                ax.plot([pos1, pos1, pos2, pos2], [y_max + h, y_max + 2*h, y_max + 2*h, y_max + h], lw=1.5, c='black')
                if p_val < significance_threshold_3: 
                    marker = "***"
                elif p_val < significance_threshold_2:
                    marker = "**"
                elif p_val < significance_threshold_1:
                    marker = "*"
                else:
                    marker = ""
                # Annotate significance
                ax.text((pos1 + pos2) / 2, y_max + 2.5*h, marker, ha='center', va='bottom', color='black', fontsize=12)
            ax.set_xticklabels(use_labels)

        elif selector == "T": 
            t_to_a_traj = []
            a_to_t_traj = []
            for i in range(len(selected_attns)): 
                this_attn = selected_attns[i]
                this_sep_frame2 = selected_sf2s[i]

                blocks = extract_attention_blocks_T(this_attn, this_sep_frame2)

                t_to_a_interp = interpolate_traj(blocks['t_to_a'], n_steps)
                a_to_t_interp = interpolate_traj(blocks['a_to_t'], n_steps)
                t_to_a_traj.append(t_to_a_interp)
                a_to_t_traj.append(a_to_t_interp)

            # Convert list of arrays into 2D NumPy arrays for easier manipulation
            group3_array = np.array(t_to_a_traj)
            group4_array = np.array(a_to_t_traj)

            target_group3 = group3_array[:, -segment_length:].flatten()
            target_group4 = group4_array[:, :segment_length].flatten()

            # Calculate the mean trajectory for each group
            means = np.array([np.mean(target_group3, axis=0), 
                            np.mean(target_group4, axis=0)])

            # Calculate the SEM for each step in both groups
            sems = np.array([sem(target_group3, axis=0),
                            sem(target_group4, axis=0)])

            # Perform statistical tests between the three specified pairs
            _, p_val_p2v_vs_v2p = ttest_ind(target_group3, target_group4)
            use_labels = legend_names[2:]
            use_colors = colors[2:]

            x = np.arange(len(use_labels))  # Label locations
            bars = ax.bar(x, means, yerr=1.96*np.array(sems), capsize=5, color=use_colors)

 
            # Mark significance directly on the bar graph
            significance_threshold_1 = 0.05
            significance_threshold_2 = 0.01
            significance_threshold_3 = 0.001
            signif_positions_pairs = [(0, 1)]  # Pairs of positions for each comparison
            p_values = [p_val_p2v_vs_v2p]

            for (pos1, pos2), p_val in zip(signif_positions_pairs, p_values):
                y_max = max(means[pos1] + 1.96*sems[pos1], means[pos2] + 1.96*sems[pos2])
                h = y_max * 0.05  # 5% above the max for drawing the line
                ax.plot([pos1, pos1, pos2, pos2], [y_max + h, y_max + 2*h, y_max + 2*h, y_max + h], lw=1.5, c='black')
                if p_val < significance_threshold_3: 
                    marker = "***"
                elif p_val < significance_threshold_2:
                    marker = "**"
                elif p_val < significance_threshold_1:
                    marker = "*"
                else:
                    marker = ""
                # Annotate significance
                ax.text((pos1 + pos2) / 2, y_max + 2.5*h, marker, ha='center', va='bottom', color='black', fontsize=12)
            ax.set_xticklabels(use_labels)

        ax.set_ylabel('Attention')
        ax.set_title(selector)
        ax.set_xticks(x)
        ax.set_ylim(0, max(means) * 1.2)  # Adjust y-axis limit for visibility

    fig.suptitle('Comparison of Foreign-Attention Trajectory')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



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
    model_type = args.model
    model_condition = args.condition
    train_name = "C_0B"
    res_save_dir = os.path.join(model_save_, f"eval-{train_name}-{ts}")

    sil_dict = {}
    model_condition_dir = os.path.join(res_save_dir, model_type, model_condition)
    assert PU.path_exist(model_condition_dir)
    this_save_dir = os.path.join(model_condition_dir, "integrated_results")
    mk(this_save_dir)

    every_attns = []
    every_sepframes1 = []
    every_sepframes2 = []
    every_phi_types = []

    for epoch in range(0, 100): 
        cat_attns = []
        cat_sepframes1 = []
        cat_sepframes2 = []
        cat_phi_types = []
        print(f"Processing {model_type} at {epoch}...")

        for run_number in range(1, 11):
            this_model_condition_dir = os.path.join(model_condition_dir, f"{run_number}")
            allres = read_result_at(this_model_condition_dir, epoch)
            cat_phi_types += allres["phi-type"]
            cat_attns += allres["attn"]
            cat_sepframes1 += allres["sep-frame1"]
            cat_sepframes2 += allres["sep-frame2"]

        plot_attention_comparison(cat_phi_types, cat_attns, cat_sepframes1, cat_sepframes2, os.path.join(this_save_dir, f"attncomp-at-{epoch}.png"))
        every_attns += cat_attns
        every_sepframes1 += cat_sepframes1
        every_sepframes2 += cat_sepframes2
        every_phi_types += cat_phi_types
    plot_attention_comparison(every_phi_types, every_attns, every_sepframes1, every_sepframes2, os.path.join(res_save_dir, f"attncomp-at-all-{model_type}-{model_condition}.png"))

    print("Done.")