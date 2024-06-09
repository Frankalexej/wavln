import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from model_model import AEPPV1, AEPPV2, AEPPV7

from model_dataset import TargetVowelDatasetBoundaryPhoneseq as ThisDataset
from model_dataset import SilenceSampler_for_TV
from C_0X_defs import *
from C_0P_run import load_dict

PLOSIVE_SUFFIX = "h"

# not using that in B, but we overwrite it here
def get_data(rec_dir, guide_path, word_guide_):
    mytrans = TheTransform(sample_rate=REC_SAMPLE_RATE, 
                        n_fft=N_FFT, n_mels=N_MELS, 
                        normalizer=Normalizer.norm_mvn, 
                        denormalizer=DeNormalizer.norm_mvn)

    st_valid = pd.read_csv(guide_path)

    mylist = load_dict()
    mymap = TokenMap(mylist)

    valid_ds = ThisDataset(rec_dir, 
                        st_valid, 
                        mapper=mymap,
                        transform=mytrans, 
                        plosive_suffix=PLOSIVE_SUFFIX)

    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=LOADER_WORKER, collate_fn=ThisDataset.collate_fn)
    return valid_loader

def get_data_both(rec_dir, t_guide_path, st_guide_path, word_guide_):
    mytrans = TheTransform(sample_rate=REC_SAMPLE_RATE, 
                        n_fft=N_FFT, n_mels=N_MELS, 
                        normalizer=Normalizer.norm_mvn, 
                        denormalizer=DeNormalizer.norm_mvn)

    st_valid = pd.read_csv(st_guide_path)
    t_valid = pd.read_csv(t_guide_path)
    t_valid["pre_startTime"] = t_valid["stop_startTime"] - SilenceSampler_for_TV().sample(len(t_valid))
    all_valid = pd.concat([t_valid, st_valid], ignore_index=True, sort=False)
    
    mylist = load_dict()
    mymap = TokenMap(mylist)

    valid_ds = ThisDataset(rec_dir, 
                        all_valid, 
                        mapper=mymap,
                        transform=mytrans, 
                        plosive_suffix=PLOSIVE_SUFFIX)

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
    
    all_z1 = []
    all_z2 = []
    all_z3 = []
    all_stop_names = []
    all_vowel_names = []
    all_sepframes1 = []
    all_sepframes2 = []
    all_attn = []
    all_pp = []
    all_phi_type = []

    for (x, x_lens, pt, sn, vn, sf1, sf2, phoneseq) in both_loader: 
        # name = name[0]

        x_mask = generate_mask_from_lengths_mat(x_lens, device=device)
        
        x = x.to(device)

        (x_hat_recon, y_hat_preds), (attn_w_recon, attn_w_preds), (zes, zqs) = model.fruitfulforward(x, x_lens, x_mask)

        z1 = zes[0].cpu().detach().numpy().squeeze()
        z2 = zes[1].cpu().detach().numpy().squeeze()
        z3 = zes[2].cpu().detach().numpy().squeeze()
        attn_w_preds = attn_w_preds.cpu().detach().numpy().squeeze()
        pp_x = y_hat_preds.cpu().detach().numpy().squeeze()

        all_z1 += [z1]
        all_z2 += [z2]
        all_z3 += [z3]
        all_attn += [attn_w_recon]
        all_attn += [attn_w_preds]
        all_pp += [pp_x]
        all_stop_names += sn
        all_vowel_names += vn
        all_sepframes1 += sf1
        all_sepframes2 += sf2
        all_phi_type += pt
    
    reshandler.res["z1"] = all_z1
    reshandler.res["z2"] = all_z2
    reshandler.res["z3"] = all_z3
    reshandler.res["sn"] = all_stop_names
    reshandler.res["vn"] = all_vowel_names
    reshandler.res["sep-frame1"] = all_sepframes1
    reshandler.res["sep-frame2"] = all_sepframes2
    reshandler.res["attn"] = all_attn
    reshandler.res["pp"] = all_pp
    reshandler.res["phi-type"] = all_phi_type
    reshandler.save()
    print(f"Results all-{stop_epoch} saved at {res_save_dir}")


    # # Plot Reconstructions
    # i = 25
    # fig, axs = plt.subplots(2, 1)
    # plot_spectrogram(all_ori[i].T, title=f"mel-spectrogram of input {all_phi_type[i]}:{all_stop_names[i]}-{all_vowel_names[i]}", ax=axs[0])
    # plot_spectrogram(all_recon[i].T, title=f"reconstructed mel-spectrogram {all_phi_type[i]}:{all_stop_names[i]}-{all_vowel_names[i]}", ax=axs[1])
    # fig.tight_layout()
    # plt.savefig(os.path.join(res_save_dir, f"recon-at-{stop_epoch}.png"))
    # plt.close()

    plot_attention_trajectory_together(all_phi_type, all_attn, all_sepframes1, all_sepframes2, os.path.join(res_save_dir, f"attntraj-at-{stop_epoch}.png"))
    return 0

def main(train_name, ts, run_number, model_type, model_save_dir, res_save_dir, guide_dir, word_guide_): 
    # Dirs
    rec_dir = train_cut_phone_
    # Check model path
    assert PU.path_exist(model_save_dir)
    assert PU.path_exist(guide_dir)

    # Load data
    st_guide_path = os.path.join(guide_dir, "ST-valid.csv")
    # single_loader = get_data(rec_dir, st_guide_path, word_guide_)
    single_loader = None
    # note that we always use the balanced data to evaluate, this is because we want the evaluation to have 
    # equal contrast, instead of having huge bias. 
    both_loader = get_data_both(rec_dir, os.path.join(guide_dir, "T-valid-sampled.csv"), st_guide_path, word_guide_)

    mylist = load_dict()
    mymap = TokenMap(mylist)
    class_dim = mymap.token_num()
    ctc_size_list = {'hid': INTER_DIM_2, 'class': class_dim}

    if model_type == "mtl":
        model = AEPPV1(enc_size_list=ENC_SIZE_LIST, 
                   dec_size_list=DEC_SIZE_LIST, 
                   ctc_decoder_size_list=ctc_size_list,
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
    elif model_type == "pp": 
        model = AEPPV2(enc_size_list=ENC_SIZE_LIST, 
                   dec_size_list=DEC_SIZE_LIST, 
                   ctc_decoder_size_list=ctc_size_list,
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
    elif model_type == "mtl-phi":
        model = AEPPV1(enc_size_list=ENC_SIZE_LIST, 
                   dec_size_list=DEC_SIZE_LIST, 
                   ctc_decoder_size_list=ctc_size_list,
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
    else: 
        raise Exception("Model type not supported! ")

    # sil_list = []
    for epoch in range(0, 50): 
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
    train_name = "C_0P"
    if not PU.path_exist(os.path.join(model_save_, f"{train_name}-{ts}-{rn}")):
        raise Exception(f"Training {train_name}-{ts}-{rn} does not exist! ")
    
    model_save_dir = os.path.join(model_save_, f"{train_name}-{ts}-{rn}", args.model, args.condition)
    guide_dir = os.path.join(model_save_, f"{train_name}-{ts}-{rn}", "guides")
    res_save_dir = os.path.join(model_save_, f"eval-{train_name}-{ts}")
    this_model_condition_dir = os.path.join(res_save_dir, args.model, args.condition, f"{rn}")
    valid_full_guide_path = os.path.join(src_, "guide_validation.csv")
    mk(this_model_condition_dir)

    main(train_name, ts, args.runnumber, args.model, model_save_dir, this_model_condition_dir, guide_dir, valid_full_guide_path)