from C_0X_defs import *

def read_result_at(res_save_dir, epoch): 
    all_handler = DictResHandler(whole_res_dir=res_save_dir, 
                                 file_prefix=f"all-{epoch}")

    all_handler.read()

    return all_handler.res


def collect_attention_epoch_trajectory(all_phi_type, all_attn, all_sepframes1, all_sepframes2, save_path, conditionlist=["ST", "T"]): 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    legend_namess = [['S-to-P', 'P-to-S', 'P-to-V', 'V-to-P'], ['#-to-P', 'P-to-#', 'P-to-V', 'V-to-P']]
    colors = ['b', 'g', 'red', 'orange']
    n_steps = 100
    segment_length = int(n_steps * 0.1)  # Calculate 20% segment length
    badcounts = {selector : 0 for selector in conditionlist}
    totalcounts = {selector : 0 for selector in conditionlist}

    res_dict = {}
    for (selector, ax, legend_names) in zip(conditionlist, [ax1, ax2], legend_namess):
        # 这个只是处理ST和T，而非循环
        resslist = []
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

            ress = np.array([target_group1, target_group2, target_group3, target_group4])
            resslist.append(ress)

        npresslist = np.array(resslist)
        npresslist = npresslist.transpose(1, 0, 2)
        res_dict[selector] = resslist

    return res_dict