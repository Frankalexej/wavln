import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from model_model import WIDAEV1, WIDAEV2
from model_dataset import TargetVowelDatasetBoundaryWord as ThisDataset
from model_dataset import WordDictionary
from C_0X_defs import *

# not using that in B, but we overwrite it here
def get_data(rec_dir, guide_path, word_guide_):
    mytrans = TheTransform(sample_rate=REC_SAMPLE_RATE, 
                        n_fft=N_FFT, n_mels=N_MELS, 
                        normalizer=Normalizer.norm_mvn, 
                        denormalizer=DeNormalizer.norm_mvn)

    st_valid = pd.read_csv(guide_path)

    if word_guide_ is not None: 
        word_guide = pd.read_csv(word_guide_)
    else: 
        word_guide = pd.read_csv(os.path.join(src_, "guide_valid.csv"))

    mymap = WordDictionary(os.path.join(src_, "unique_words_list.dict"))

    valid_ds = ThisDataset(rec_dir, 
                        st_valid, 
                        select=word_guide, 
                        mapper=mymap,
                        transform=mytrans)

    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=LOADER_WORKER, collate_fn=ThisDataset.collate_fn)
    return valid_loader

def get_data_both(rec_dir, t_guide_path, st_guide_path, word_guide_):
    mytrans = TheTransform(sample_rate=REC_SAMPLE_RATE, 
                        n_fft=N_FFT, n_mels=N_MELS, 
                        normalizer=Normalizer.norm_mvn, 
                        denormalizer=DeNormalizer.norm_mvn)

    st_valid = pd.read_csv(st_guide_path)
    t_valid = pd.read_csv(t_guide_path)
    t_valid["pre_startTime"] = t_valid['stop_startTime']
    all_valid = pd.concat([t_valid, st_valid], ignore_index=True, sort=False)

    if word_guide_ is not None: 
        word_guide = pd.read_csv(word_guide_)
    else: 
        word_guide = pd.read_csv(os.path.join(src_, "guide_valid.csv"))

    mymap = WordDictionary(os.path.join(src_, "unique_words_list.dict"))

    valid_ds = ThisDataset(rec_dir, 
                        all_valid, 
                        select=word_guide,
                        mapper=mymap,
                        transform=mytrans)

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
    all_ze = []
    all_zq = []
    all_stop_names = []
    all_vowel_names = []
    all_sepframes1 = []
    all_sepframes2 = []
    all_attn = []
    all_recon = []
    all_ori = []
    all_phi_type = []

    for (x, x_lens, pt, sn, vn, sf1, sf2, word) in both_loader: 
        # name = name[0]

        x_mask = generate_mask_from_lengths_mat(x_lens, device=device)
        
        x = x.to(device)
        word = torch.tensor(word, dtype=torch.long).to(device)

        x_hat, attn_w, (ze, zq) = model(x, x_lens, x_mask, word)

        ze = ze.cpu().detach().numpy().squeeze()
        zq = zq.cpu().detach().numpy().squeeze()
        attn_w = attn_w.cpu().detach().numpy().squeeze()
        
        recon_x = x_hat.cpu().detach().numpy().squeeze()
        ori_x = x.cpu().detach().numpy().squeeze()

        all_ze += [ze]
        all_zq += [zq]
        all_attn += [attn_w]
        all_recon += [recon_x]
        all_ori += [ori_x]
        all_stop_names += sn
        all_vowel_names += vn
        all_sepframes1 += sf1
        all_sepframes2 += sf2
        all_phi_type += pt
    
    reshandler.res["ze"] = all_ze
    reshandler.res["zq"] = all_zq
    reshandler.res["sn"] = all_stop_names
    reshandler.res["vn"] = all_vowel_names
    reshandler.res["sep-frame1"] = all_sepframes1
    reshandler.res["sep-frame2"] = all_sepframes2
    reshandler.res["attn"] = all_attn
    reshandler.res["recon"] = all_recon
    reshandler.res["ori"] = all_ori
    reshandler.res["phi-type"] = all_phi_type
    reshandler.save()
    print(f"Results all-{stop_epoch} saved at {res_save_dir}")


    # Plot Reconstructions
    i = 25
    fig, axs = plt.subplots(2, 1)
    plot_spectrogram(all_ori[i].T, title=f"mel-spectrogram of input {all_phi_type[i]}:{all_stop_names[i]}-{all_vowel_names[i]}", ax=axs[0])
    plot_spectrogram(all_recon[i].T, title=f"reconstructed mel-spectrogram {all_phi_type[i]}:{all_stop_names[i]}-{all_vowel_names[i]}", ax=axs[1])
    fig.tight_layout()
    plt.savefig(os.path.join(res_save_dir, f"recon-at-{stop_epoch}.png"))
    plt.close()

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
    single_loader = get_data(rec_dir, st_guide_path, word_guide_)
    # note that we always use the balanced data to evaluate, this is because we want the evaluation to have 
    # equal contrast, instead of having huge bias. 
    both_loader = get_data_both(rec_dir, os.path.join(guide_dir, "T-valid-sampled.csv"), st_guide_path, word_guide_)

    mapper = WordDictionary(os.path.join(src_, "unique_words_list.dict"))
    embedding_dim = mapper.token_num()
    if model_type == "ae":
        model = WIDAEV1(enc_size_list=ENC_SIZE_LIST, 
                   dec_size_list=DEC_SIZE_LIST, 
                   embedding_dim=embedding_dim, 
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
    elif model_type == "vqvae":
        model = WIDAEV1(enc_size_list=ENC_SIZE_LIST, 
                   dec_size_list=DEC_SIZE_LIST, 
                   embedding_dim=embedding_dim, 
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
    elif model_type == "aefixemb":
        model = WIDAEV2(enc_size_list=ENC_SIZE_LIST, 
                   dec_size_list=DEC_SIZE_LIST, 
                   embedding_dim=embedding_dim, 
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
    else:
        model = WIDAEV1(enc_size_list=ENC_SIZE_LIST, 
                   dec_size_list=DEC_SIZE_LIST, 
                   embedding_dim=embedding_dim, 
                   num_layers=NUM_LAYERS, dropout=DROPOUT)

    # sil_list = []
    for epoch in range(0, 100): 
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
    train_name = "C_0D"
    if not PU.path_exist(os.path.join(model_save_, f"{train_name}-{ts}-{rn}")):
        raise Exception(f"Training {train_name}-{ts}-{rn} does not exist! ")
    
    model_save_dir = os.path.join(model_save_, f"{train_name}-{ts}-{rn}", args.model, args.condition)
    guide_dir = os.path.join(model_save_, f"{train_name}-{ts}-{rn}", "guides")
    res_save_dir = os.path.join(model_save_, f"eval-{train_name}-{ts}")
    this_model_condition_dir = os.path.join(res_save_dir, args.model, args.condition, f"{rn}")
    valid_full_guide_path = os.path.join(src_, "guide_validation.csv")
    mk(this_model_condition_dir)

    main(train_name, ts, args.runnumber, args.model, model_save_dir, this_model_condition_dir, guide_dir, valid_full_guide_path)