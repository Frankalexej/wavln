import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from C_0X_defs import *

def read_result_at(res_save_dir, epoch): 
    all_handler = DictResHandler(whole_res_dir=res_save_dir, 
                                 file_prefix=f"all-{epoch}")

    all_handler.read()

    return all_handler.res


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
    train_name = "C_0R"
    res_save_dir = os.path.join(model_save_, f"eval-{train_name}-{ts}")

    sil_dict = {}
    model_condition_dir = os.path.join(res_save_dir, model_type, model_condition)
    print(model_condition_dir)
    assert PU.path_exist(model_condition_dir)
    this_save_dir = os.path.join(model_condition_dir, "integrated_results")
    mk(this_save_dir)

    every_attns = []
    every_attns_pp = []
    every_sepframes1 = []
    every_sepframes2 = []
    every_phi_types = []

    # this is added because some runs failed to do any reconstruction. 
    learned_runs = [1, 2, 3, 4, 5]
    string_learned_runs = [str(num) for num in learned_runs]
    strseq_learned_runs = "".join(string_learned_runs)

    for epoch in range(0, 100): 
        cat_attns = []
        cat_attns_pp = []
        cat_sepframes1 = []
        cat_sepframes2 = []
        cat_phi_types = []
        print(f"Processing {model_type} at {epoch}...")

        for run_number in learned_runs:
            this_model_condition_dir = os.path.join(model_condition_dir, f"{run_number}")
            if not PU.path_exist(this_model_condition_dir): 
                print(f"Run {run_number} does not exist.")
                continue
            allres = read_result_at(this_model_condition_dir, epoch)
            cat_phi_types += allres["phi-type"]
            cat_attns += allres["attn"]
            # cat_attns_pp += allres["attn-pp"]
            cat_sepframes1 += allres["sep-frame1"]
            cat_sepframes2 += allres["sep-frame2"]

        # plot_attention_trajectory_together(cat_phi_types, cat_attns, cat_sepframes1, cat_sepframes2, os.path.join(this_save_dir, f"attntraj-at-{epoch}.png"))
        every_attns += cat_attns
        # every_attns_pp += cat_attns_pp
        every_sepframes1 += cat_sepframes1
        every_sepframes2 += cat_sepframes2
        every_phi_types += cat_phi_types
    plot_attention_trajectory_together(every_phi_types, every_attns, every_sepframes1, every_sepframes2, os.path.join(res_save_dir, f"attntraj-at-all-{model_type}-{model_condition}-{strseq_learned_runs}.png"))
    # plot_attention_trajectory_together(every_phi_types, every_attns_pp, every_sepframes1, every_sepframes2, os.path.join(res_save_dir, f"attnpptraj-at-all-{model_type}-{model_condition}-{strseq_learned_runs}.png"))

    print("Done.")