from pdb import run
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from model_model import AEPPV1, AEPPV2, AEPPV4, AEPPV9

from C_0X_defs import *
from model_dataset import TargetVowelDatasetBoundaryPhoneseqSpeakerGender as ThisDataset
from model_dataset import MelSpecTransformDB as TheTransform
from model_dataset import SilenceSampler_for_TV

BATCH_SIZE = 1
INPUT_DIM = 64
OUTPUT_DIM = 64 
INTER_DIM_0 = 32
INTER_DIM_1 = 16
INTER_DIM_2 = 8
ENC_SIZE_LIST = [INPUT_DIM, INTER_DIM_0, INTER_DIM_1, INTER_DIM_2]
DEC_SIZE_LIST = [OUTPUT_DIM, INTER_DIM_0, INTER_DIM_1, INTER_DIM_2]
DROPOUT = 0.5
NUM_LAYERS = 2
EMBEDDING_DIM = 128
REC_SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 64
LOADER_WORKER = 32

# not using that in B, but we overwrite it here
def get_data(rec_dir, guide_path, word_guide_):
    mytrans = TheTransform(sample_rate=REC_SAMPLE_RATE, 
                        n_fft=N_FFT, n_mels=N_MELS, 
                        normalizer=Normalizer.norm_mvn, 
                        denormalizer=DeNormalizer.norm_mvn)

    st_valid = pd.read_csv(guide_path)

    # Load TokenMap to map the phoneme to the index
    with open(os.path.join(src_, "no-stress-seg.dict"), "rb") as file:
        # Load the object from the file
        mylist = pickle.load(file)
        mylist = ["BLANK"] + mylist
        mylist = mylist + ["SIL"]

    # Now you can use the loaded object
    mymap = TokenMap(mylist)

    valid_ds = ThisDataset(rec_dir, 
                        st_valid, 
                        mapper=mymap,
                        transform=mytrans, 
                        hop_length=N_FFT//2)

    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=LOADER_WORKER, collate_fn=ThisDataset.collate_fn)
    return valid_loader

def get_data_both(rec_dir, t_guide_path, st_guide_path, word_guide_, 
                  noise_controls={"fixlength": False, "amplitude_scale": 0.01}):
    mytrans = TheTransform(sample_rate=REC_SAMPLE_RATE, 
                        n_fft=N_FFT, n_mels=N_MELS, 
                        normalizer=Normalizer.norm_mvn, 
                        denormalizer=DeNormalizer.norm_mvn)

    st_valid = pd.read_csv(st_guide_path)
    t_valid = pd.read_csv(t_guide_path)
    # now st also has noise, so we need to sample silence for both
    # st_valid["pre_startTime"] = st_valid["stop_startTime"] - SilenceSampler_for_TV(fixlength=noise_controls["fixlength"]).sample(len(st_valid))
    t_valid["pre_startTime"] = t_valid["stop_startTime"] - SilenceSampler_for_TV(fixlength=noise_controls["fixlength"]).sample(len(t_valid))
    all_valid = pd.concat([t_valid, st_valid], ignore_index=True, sort=False)
    # all_valid.to_csv("all_valid.csv", index=False)
    # raise Exception("Stop here")

    # Load TokenMap to map the phoneme to the index
    with open(os.path.join(src_, "no-stress-seg.dict"), "rb") as file:
        # Load the object from the file
        mylist = pickle.load(file)
        mylist = ["BLANK"] + mylist
        mylist = mylist + ["SIL"]

    # Now you can use the loaded object
    mymap = TokenMap(mylist)

    valid_ds = ThisDataset(rec_dir, 
                        all_valid, 
                        mapper=mymap,
                        transform=mytrans, 
                        hop_length=N_FFT//2, 
                        noise_amplitude_scale=noise_controls["amplitude_scale"], 
                        speaker_meta_path=os.path.join(src_, "speakers.csv"))

    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=LOADER_WORKER, collate_fn=ThisDataset.collate_fn)
    return valid_loader

def run_one_epoch(model, single_loader, both_loader, model_save_dir, stop_epoch, res_save_dir, hiddim): 
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recon_loss = nn.MSELoss(reduction='none')
    masked_recon_loss = MaskedLoss(recon_loss)
    model_loss = masked_recon_loss

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # if not pre-epoch, then load model
    # if stop_epoch == 999: 
    #     initialize_model(model)
    # else: 
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
    # all_attn_pp = []
    all_recon = []
    all_attnout = []
    all_ori = []
    all_phi_type = []
    all_other_hid_outs = {
        "enc-lin1": [], 
        "dec-lin1": [],
        "enc-rnn1-f": [],
        "dec-rnn1-f": [],
        "enc-rnn1-b": [],
        "enc-rnn2-f": [],
        "dec-rnn2-f": [],
        "enc-rnn2-b": [],
    }
    all_sid = []
    all_gender = []

    for (x, x_lens, pt, sn, vn, sf1, sf2, phoneseq, sid, gender) in both_loader: 
        # name = name[0]

        x_mask = generate_mask_from_lengths_mat(x_lens, device=device)
        
        x = x.to(device)

        (x_hat_recon, x_hat_attn), (attn_w_recon, attn_w_preds), (ze, zq), (enc_hid_outs, dec_hid_outs) = model.attn_forward(x, x_lens, x_mask)

        ze = ze.cpu().detach().numpy().squeeze()
        zq = zq.cpu().detach().numpy().squeeze()
        attn_w_recon = attn_w_recon.cpu().detach().numpy().squeeze()
        # attn_w_preds = attn_w_preds.cpu().detach().numpy().squeeze()
        
        recon_x = x_hat_recon.cpu().detach().numpy().squeeze()
        attn_out = x_hat_attn.cpu().detach().numpy().squeeze()
        ori_x = x.cpu().detach().numpy().squeeze()


        if np.any(np.isnan(attn_w_recon)): 
            print(f"NaN detected at {stop_epoch}!")
            continue

        all_ze += [ze]
        all_zq += [zq]
        all_attn += [attn_w_recon]
        # all_attn_pp += [attn_w_preds]
        all_recon += [recon_x]
        all_attnout += [attn_out]
        all_ori += [ori_x]
        all_stop_names += sn
        all_vowel_names += vn
        all_sepframes1 += sf1
        all_sepframes2 += sf2
        all_phi_type += pt
        all_sid += sid
        all_gender += gender

        # deal with all other hidden outputs
        all_other_hid_outs["enc-lin1"] += [enc_hid_outs[0].cpu().detach().numpy().squeeze()]
        all_other_hid_outs["dec-lin1"] += [dec_hid_outs[0].cpu().detach().numpy().squeeze()]
        all_other_hid_outs["enc-rnn1-f"] += [enc_hid_outs[1][:, :, :hiddim].cpu().detach().numpy().squeeze()]
        all_other_hid_outs["dec-rnn1-f"] += [dec_hid_outs[1].cpu().detach().numpy().squeeze()]
        all_other_hid_outs["enc-rnn1-b"] += [enc_hid_outs[1][:, :, hiddim:].cpu().detach().numpy().squeeze()]
        all_other_hid_outs["enc-rnn2-f"] += [enc_hid_outs[2][:, :, :hiddim].cpu().detach().numpy().squeeze()]
        all_other_hid_outs["dec-rnn2-f"] += [dec_hid_outs[2].cpu().detach().numpy().squeeze()]
        all_other_hid_outs["enc-rnn2-b"] += [enc_hid_outs[2][:, :, hiddim:].cpu().detach().numpy().squeeze()]
        """
        At the moment I believe that the LSTM's output concatenation is done 
        is the way such that when I separate the outputs from the middle, 
        the first half is forward and the second half is backward; and that 
        the same index refers to the same time step in terms of INPUT SEQUENCE. 
        This is not the WORKING PROGRESS SEQUENCE, which refers to the order 
        that the LSTM takes the items in; instead, I think it referst to the 
        timesteps in the input sequence as we feed into the LSTM. 
        """
    
    reshandler.res["ze"] = all_ze
    reshandler.res["zq"] = all_attnout  # 因为没用，所以用这个存
    reshandler.res["sn"] = all_stop_names
    reshandler.res["vn"] = all_vowel_names
    reshandler.res["sep-frame1"] = all_sepframes1
    reshandler.res["sep-frame2"] = all_sepframes2
    reshandler.res["attn"] = all_attn
    # reshandler.res["attn-pp"] = all_attn_pp
    reshandler.res["recon"] = all_recon
    # reshandler.res["attnout"] = all_attnout
    reshandler.res["ori"] = all_ori
    reshandler.res["phi-type"] = all_phi_type
    reshandler.res["other-hid-outs"] = all_other_hid_outs
    reshandler.res["sid"] = all_sid
    reshandler.res["gender"] = all_gender
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

    """Commented out because sometimes due to drawing problems some runs could not be plotted (I guess it's because some have zero attention)"""
    try: 
        plot_attention_trajectory_together(all_phi_type, all_attn, all_sepframes1, all_sepframes2, os.path.join(res_save_dir, f"attntraj-at-{stop_epoch}.png"), 
                                        conditionlist=["T", "ST"])
    except Exception as e: 
        print(f"Error in plotting attention trajectory: {e}")
    return 0

def main(train_name, ts, run_number, model_type, model_save_dir, res_save_dir, guide_dir, word_guide_, 
         noise_controls={"fixlength": False, "amplitude_scale": 0.01}): 
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
    both_loader = get_data_both(rec_dir, os.path.join(guide_dir, "T-valid-sampled.csv"), 
                                st_guide_path, word_guide_, 
                                noise_controls=noise_controls)

    # Load TokenMap to map the phoneme to the index
    with open(os.path.join(src_, "no-stress-seg.dict"), "rb") as file:
        # Load the object from the file
        mylist = pickle.load(file)
        mylist = ["BLANK"] + mylist
        mylist = mylist + ["SIL"]   # this is to fit STV vs #TV

    # Now you can use the loaded object
    mymap = TokenMap(mylist)
    class_dim = mymap.token_num()
    ctc_size_list = {'hid': INTER_DIM_2, 'class': class_dim}

    if model_type in ["recon4-phi", "recon8-phi", "recon16-phi", "recon32-phi", 
                        "recon48-phi", "recon64-phi", "recon96-phi", "recon128-phi"]: 
        hiddim = int(model_type.split("-")[0].replace("recon", "")) # get hidden dimension from model_type
        enc_list = [INPUT_DIM, INTER_DIM_0, INTER_DIM_1, hiddim]
        dec_list = [OUTPUT_DIM, INTER_DIM_0, INTER_DIM_1, hiddim]
        model = AEPPV9(enc_size_list=enc_list, 
                   dec_size_list=dec_list, 
                   ctc_decoder_size_list=ctc_size_list,
                   num_layers=NUM_LAYERS, dropout=DROPOUT)
    else: 
        raise Exception("Model type not supported! ")
    # Run pre-training epoch check (999) 
    # run_one_epoch(model, single_loader, both_loader, model_save_dir, 999, res_save_dir)

    fixer_starting_epoch = 0
    # if model_type == "recon128-phi":
    #     fixer_starting_epoch = 87
    # hiddim = int(model_type.split("-")[0].replace("recon", ""))
    # if hiddim == 48: 
    #     fixer_starting_epoch = 45
    # elif hiddim == 96: 
    #     fixer_starting_epoch = 61
    # elif hiddim == 128: 
    #     fixer_starting_epoch = 79
    # else: 
    #     raise Exception("Model type not supported in fixing! ")
    # sil_list = []
    for epoch in range(fixer_starting_epoch, 101): # 100 -> 101, because 0 is pre-training epoch 
        run_one_epoch(model, single_loader, both_loader, model_save_dir, epoch, res_save_dir, hiddim=hiddim)
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
    model_type = args.model

    rn = int(rn)

    train_name = "C_0Tf"
    if not PU.path_exist(os.path.join(model_save_, f"{train_name}-{ts}-{rn}")):
        raise Exception(f"Training {train_name}-{ts}-{rn} does not exist! ")
    
    model_save_dir = os.path.join(model_save_, f"{train_name}-{ts}-{rn}", args.model, args.condition)
    guide_dir = os.path.join(model_save_, f"{train_name}-{ts}-{rn}", "guides")
    res_save_dir = os.path.join(model_save_, f"eval-{train_name}-{ts}")
    this_model_condition_dir = os.path.join(res_save_dir, args.model, args.condition, f"{rn}")
    valid_full_guide_path = os.path.join(src_, "guide_validation.csv")
    mk(this_model_condition_dir)

    main(train_name, ts, rn, model_type, model_save_dir, this_model_condition_dir, guide_dir, valid_full_guide_path, 
         noise_controls={"fixlength": False, "amplitude_scale": 0.004})