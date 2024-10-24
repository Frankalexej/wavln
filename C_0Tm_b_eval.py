from pdb import run
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from model_model import MultiBlockV1

from C_0X_defs import *
from model_dataset import TargetVowelDatasetBoundaryPhoneseqSpeakerGenderManualNorm as ThisDataset
from model_dataset import MelSpecTransformDBNoNorm as TheTransform
from model_dataset import NormalizerMVNManual, TokenMap
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
NUM_BLOCKS = 3
EMBEDDING_DIM = 128
REC_SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 64
LOADER_WORKER = 32

# not using that in B, but we overwrite it here
def get_data(rec_dir, guide_path, word_guide_):
    raise NotImplementedError("This function should not be used in this thread. ")

def get_data_both(rec_dir, t_guide_path, st_guide_path, word_guide_, 
                  noise_controls={"fixlength": False, "amplitude_scale": 0.01}, 
                  mv_config=None, 
                  st_has_pre=True): 
    mytrans = TheTransform(sample_rate=REC_SAMPLE_RATE, 
                        n_fft=N_FFT, n_mels=N_MELS)
    mynorm = NormalizerMVNManual()

    st_valid = pd.read_csv(st_guide_path)
    t_valid = pd.read_csv(t_guide_path)
    t_valid["pre_startTime"] = t_valid["stop_startTime"] - SilenceSampler_for_TV(fixlength=noise_controls["fixlength"]).sample(len(t_valid))
    if not st_has_pre: 
        # now st also has noise, so we need to sample silence for both
        st_valid["pre_startTime"] = st_valid["stop_startTime"] - SilenceSampler_for_TV(fixlength=noise_controls["fixlength"]).sample(len(st_valid))
    all_valid = pd.concat([t_valid, st_valid], ignore_index=True, sort=False)

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
                        normalizer=mynorm, 
                        hop_length=N_FFT//2, 
                        noise_amplitude_scale=noise_controls["amplitude_scale"], 
                        speaker_meta_path=os.path.join(src_, "speakers.csv"), 
                        mv_config=mv_config)

    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=LOADER_WORKER, collate_fn=ThisDataset.collate_fn)
    return valid_loader

def run_one_epoch(model, single_loader, both_loader, model_save_dir, stop_epoch, res_save_dir, hiddim, 
                  condition_list=["T", "ST"]): 
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

    all_stop_names = []
    all_vowel_names = []
    all_sepframes1 = []
    all_sepframes2 = []
    all_attn = {f"{i+1}" : [] for i in range(NUM_BLOCKS)}
    all_recon = []
    all_ori = []
    all_phi_type = []
    all_sid = []
    all_gender = []
    all_hidlayer_outs = {}
    for idx in range(NUM_BLOCKS): 
        all_hidlayer_outs[f"hidrep-{idx+1}"] = []
        all_hidlayer_outs[f"attnout-{idx+1}"] = []
        all_hidlayer_outs[f"decrep-{idx+1}"] = []
        for jdx in range(NUM_LAYERS): 
            all_hidlayer_outs[f"encrnn-{idx+1}-{jdx+1}-f"] = []
            all_hidlayer_outs[f"encrnn-{idx+1}-{jdx+1}-b"] = []
            all_hidlayer_outs[f"decrnn-{idx+1}-{jdx+1}-f"] = []

    for (x, x_lens, pt, sn, vn, sf1, sf2, phoneseq, sid, gender) in both_loader: 
        x_mask = generate_mask_from_lengths_mat(x_lens, device=device)
        x = x.to(device)

        x_recon, attn_ws, hidlayer_outs = model.run_and_out(x, x_lens, x_mask)

        # original input and reconstructed output
        x_recon = x_recon.cpu().detach().numpy().squeeze()
        ori_x = x.cpu().detach().numpy().squeeze()

        # Data with blocks
        for i in range(NUM_BLOCKS): 
            all_attn[f"{i+1}"] += [attn_ws[i].cpu().detach().numpy().squeeze()]

        all_recon += [x_recon]
        all_ori += [ori_x]
        all_stop_names += sn
        all_vowel_names += vn
        all_sepframes1 += sf1
        all_sepframes2 += sf2
        all_phi_type += pt
        all_sid += sid
        all_gender += gender

        for layername, layerout in hidlayer_outs.items(): 
            all_hidlayer_outs[layername] += [layerout.cpu().detach().numpy().squeeze()]
        """
        At the moment I believe that the LSTM's output concatenation is done 
        is the way such that when I separate the outputs from the middle, 
        the first half is forward and the second half is backward; and that 
        the same index refers to the same time step in terms of INPUT SEQUENCE. 
        This is not the WORKING PROGRESS SEQUENCE, which refers to the order 
        that the LSTM takes the items in; instead, I think it referst to the 
        timesteps in the input sequence as we feed into the LSTM. 
        """
    
    # Save results
    reshandler.res["sn"] = all_stop_names
    reshandler.res["vn"] = all_vowel_names
    reshandler.res["sep-frame1"] = all_sepframes1
    reshandler.res["sep-frame2"] = all_sepframes2
    reshandler.res["attn"] = all_attn
    reshandler.res["recon"] = all_recon
    reshandler.res["ori"] = all_ori
    reshandler.res["phi-type"] = all_phi_type
    reshandler.res["hidlayer-outs"] = all_hidlayer_outs
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
    for attnname, attndatas in all_attn.items(): 
        try: 
            plot_attention_trajectory_together(all_phi_type, attndatas, all_sepframes1, all_sepframes2, os.path.join(res_save_dir, f"attntraj-block-{attnname}-at-{stop_epoch}.png"), 
                                            conditionlist=condition_list)
        except Exception as e: 
            print(f"Error in plotting attention trajectory: {e}")
    return 0

def main(train_name, ts, run_number, model_type, model_save_dir, res_save_dir, guide_dir, word_guide_, 
         nameset={"larger": "T", "smaller": "ST"}, 
         noise_controls={"fixlength": False, "amplitude_scale": 0.01}): 
    # Dirs
    rec_dir = train_cut_phone_
    # Check model path
    assert PU.path_exist(model_save_dir)
    assert PU.path_exist(guide_dir)

    # Load data
    st_guide_path = os.path.join(guide_dir, f"{nameset['smaller']}-valid.csv")
    t_guide_path = os.path.join(guide_dir, f"{nameset['larger']}-valid-sampled.csv")

    st_has_pre = True if nameset["smaller"] == "ST" else False
    single_loader = None
    # note that we always use the balanced data to evaluate, this is because we want the evaluation to have 
    # equal contrast, instead of having huge bias. 

    # Load MV_config
    with open(os.path.join(src_, "mv_config.pkl"), "rb") as file: 
        mv_config = pickle.load(file)

    both_loader = get_data_both(rec_dir, t_guide_path, 
                                st_guide_path, word_guide_, 
                                noise_controls=noise_controls, 
                                mv_config=mv_config, 
                                st_has_pre=st_has_pre)

    # Load TokenMap to map the phoneme to the index
    with open(os.path.join(src_, "no-stress-seg.dict"), "rb") as file:
        # Load the object from the file
        mylist = pickle.load(file)
        mylist = ["BLANK"] + mylist
        mylist = mylist + ["SIL"]   # this is to fit STV vs #TV

    if model_type in ["recon4-phi", "recon8-phi", "recon16-phi", "recon32-phi", 
                        "recon48-phi", "recon64-phi", "recon96-phi", "recon128-phi"]: 
        hiddim = int(model_type.split("-")[0].replace("recon", "")) # get hidden dimension from model_type
        enc_list = {"in": INPUT_DIM, "hid": hiddim}
        dec_list = {"hid": hiddim, "out": OUTPUT_DIM}
        model = MultiBlockV1(enc_size_list=enc_list, 
                   dec_size_list=dec_list, 
                   num_layers=NUM_LAYERS, dropout=DROPOUT, 
                   num_blocks=NUM_BLOCKS, residual=True)    # NOTE: residual is temporarily set here. 
    else: 
        raise Exception("Model type not supported! ")
    # Run pre-training epoch check (999) 
    # run_one_epoch(model, single_loader, both_loader, model_save_dir, 999, res_save_dir)

    fixer_starting_epoch = 0
    for epoch in range(fixer_starting_epoch, 101): # 100 -> 101, because 0 is pre-training epoch 
        run_one_epoch(model, single_loader, both_loader, model_save_dir, epoch, res_save_dir, hiddim=hiddim, 
                      condition_list=[nameset["larger"], nameset["smaller"]])
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

    train_name = "C_0Tm"
    if not PU.path_exist(os.path.join(model_save_, f"{train_name}-{ts}-{rn}")):
        raise Exception(f"Training {train_name}-{ts}-{rn} does not exist! ")
    
    model_save_dir = os.path.join(model_save_, f"{train_name}-{ts}-{rn}", args.model, args.condition)
    guide_dir = os.path.join(model_save_, f"{train_name}-{ts}-{rn}", "guides")
    res_save_dir = os.path.join(model_save_, f"eval-{train_name}-{ts}")
    this_model_condition_dir = os.path.join(res_save_dir, args.model, args.condition, f"{rn}")
    valid_full_guide_path = os.path.join(src_, "guide_validation.csv")
    mk(this_model_condition_dir)

    main(train_name, ts, rn, model_type, model_save_dir, this_model_condition_dir, guide_dir, valid_full_guide_path, 
         nameset={"larger": "T", "smaller": "ST"}, noise_controls={"fixlength": False, "amplitude_scale": 0.004})