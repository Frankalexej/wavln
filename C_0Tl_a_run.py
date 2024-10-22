"""
R is completely the same as E, but we delete PP, and will mainly use the phenomenon dataset to train the model. 

T is replicating R. But with 2 layers only. 

TA is td train. The model learns on TV and DV sequences.  
In this thread, we will make the codes more modularized and general. 

TC is running sPV/PV again, but deleting s and replace that with #, i.e. #PPV/#PV (PP means unaspirated P). 

TE is running PV/P'V again, but with AEPPV9. Also, I would tune down the learning rate to 5e-4 and make noise quieter.

TF is running sPV/PV, but with AEPPV9. Also, I would tune down the learning rate to 1e-4 and make noise quieter. Same as TE. 

TJ is new normalization method with sPVPV. 
"""
######################### Libs #########################
from tkinter.font import names
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
# import summary
from model_model import AEPPV1, AEPPV2, AEPPV4, AEPPV11
from model_dataset import DS_Tools
from model_dataset import TargetVowelDatasetManualNorm as TestDataset
from model_dataset import NormalizerMVNManual, DeNormalizer, TokenMap, WordDictionary
from model_dataset import MelSpecTransformDBNoNorm as TheTransform
from paths import *
from misc_my_utils import *
from misc_recorder import *
from model_loss import *
from model_padding import generate_mask_from_lengths_mat

BATCH_SIZE = 512
INPUT_DIM = 64
OUTPUT_DIM = 64 
INTER_DIM_0 = 32
INTER_DIM_1 = 16
INTER_DIM_2 = 8
ENC_SIZE_LIST = [INPUT_DIM, INTER_DIM_0, INTER_DIM_1, INTER_DIM_2]
DEC_SIZE_LIST = [OUTPUT_DIM, INTER_DIM_0, INTER_DIM_1, INTER_DIM_2]
DROPOUT = 0.5
NUM_LAYERS = 2
NUM_BIG_LAYERS = 3
EMBEDDING_DIM = 128
REC_SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 64
LOADER_WORKER = 32


def random_sample_by_speaker(larger, smaller, valid_proportion=0.2): 
    # sample speakers, tg and stg use the same, because they must have the same speaker choice 
    speakerlist = smaller["speaker"].unique()
    valid_size = int(len(speakerlist) * valid_proportion)
    sampled_indices = np.random.choice(len(speakerlist), size=valid_size, replace=False)
    valid_speakers = speakerlist[sampled_indices]

    t_smaller = smaller[~smaller["speaker"].isin(valid_speakers)]   # training set
    v_smaller = smaller[smaller["speaker"].isin(valid_speakers)]    # validation set

    t_larger = larger[~larger["speaker"].isin(valid_speakers)]
    v_larger = larger[larger["speaker"].isin(valid_speakers)]

    t_s_larger = t_larger.sample(n=len(t_smaller))
    v_s_larger = v_larger.sample(n=len(v_smaller))

    return (t_smaller, v_smaller), (t_larger, v_larger), (t_s_larger, v_s_larger)


def generate_separation(larger_path, smaller_path, target_path, nameset={"larger": "T", "smaller": "ST"}): 
    # in fact, when we use unbalanced, it is no much difference between larger and smaller. 
    # only when want to use balanced, we need to distinguish the order. 
    larger_guide = pd.read_csv(larger_path)
    smaller_guide = pd.read_csv(smaller_path)
    (training_smaller, valid_smaller), (training_larger, valid_larger), (training_sampled_larger, valid_sampled_larger) = random_sample_by_speaker(larger_guide, smaller_guide)
    # note that st set is always the same, because it will not undergo any sampling, it is the minor one. 
    training_smaller.to_csv(os.path.join(target_path, f"{nameset['smaller']}-train.csv"), index=False)
    valid_smaller.to_csv(os.path.join(target_path, f"{nameset['smaller']}-valid.csv"), index=False)
    training_larger.to_csv(os.path.join(target_path, f"{nameset['larger']}-train.csv"), index=False)
    valid_larger.to_csv(os.path.join(target_path, f"{nameset['larger']}-valid.csv"), index=False)
    training_sampled_larger.to_csv(os.path.join(target_path, f"{nameset['larger']}-train-sampled.csv"), index=False)
    valid_sampled_larger.to_csv(os.path.join(target_path, f"{nameset['larger']}-valid-sampled.csv"), index=False)
    return 

def load_data_general(dataset, rec_dir, target_path, load="train", select=0.3, sampled=True, batch_size=1):
    raise NotImplementedError("This function should not be used in this thread. ")

def load_data_phenomenon(dataset, rec_dir, target_path, load="train", select="both", 
                         sampled=True, batch_size=1, nameset={"larger": "T", "smaller": "ST"}, 
                         noise_controls={"fixlength": False, "amplitude_scale": 0.01}, 
                         mv_config=None):
    if sampled: 
        sample_suffix = "-sampled"
    else:
        sample_suffix = ""

    if select == "both":
        t_set = pd.read_csv(os.path.join(target_path, f"{nameset['larger']}-{load}{sample_suffix}.csv"))
        st_set = pd.read_csv(os.path.join(target_path, f"{nameset['smaller']}-{load}.csv"))
        integrated = pd.concat([t_set, st_set], ignore_index=True, sort=False)
    elif select == {nameset['larger']}:
        integrated = pd.read_csv(os.path.join(target_path, f"{nameset['larger']}-{load}{sample_suffix}.csv"))
    elif select == {nameset['smaller']}:
        integrated = pd.read_csv(os.path.join(target_path, f"{nameset['smaller']}-{load}.csv"))
    else: 
        raise ValueError("select must be either both, T or ST")

    integrated = integrated.sample(frac=1).reset_index(drop=True)

    mytrans = TheTransform(sample_rate=REC_SAMPLE_RATE, 
                        n_fft=N_FFT, n_mels=N_MELS)
    mynorm = NormalizerMVNManual()
    
    # Load TokenMap to map the phoneme to the index
    with open(os.path.join(src_, "no-stress-seg.dict"), "rb") as file:
        # Load the object from the file
        mylist = pickle.load(file)
        mylist = ["BLANK"] + mylist
        mylist = mylist + ["SIL"]

    # Now you can use the loaded object
    mymap = TokenMap(mylist)

    ds = dataset(rec_dir, 
                        integrated, 
                        mapper=mymap,
                        transform=mytrans, 
                        normalizer=mynorm, 
                        noise_fixlength=noise_controls["fixlength"], 
                        noise_amplitude_scale=noise_controls["amplitude_scale"], 
                        mv_config=mv_config)

    use_shuffle = True if load == "train" else False
    loader = DataLoader(ds, batch_size=batch_size, shuffle=use_shuffle, num_workers=LOADER_WORKER, collate_fn=dataset.collate_fn)
    return loader


def draw_learning_curve_and_accuracy(losses, recons, embeddings, commitments, start=0, end=100, save_name=""):
    train_losses, valid_losses, onlyST_valid_losses = losses
    train_recon_losses, valid_recon_losses, onlyST_valid_recon_losses = recons
    train_embedding_losses, valid_embedding_losses, onlyST_valid_embedding_losses = embeddings
    train_commitment_losses, valid_commitment_losses, onlyST_valid_commitment_losses = commitments

    start, end = 0, 100
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 4))
    # Plot Loss on the left subplot
    ax1.plot(train_losses.get()[start:end], label='Train')
    ax1.plot(valid_losses.get()[start:end], label='Valid')
    ax1.plot(onlyST_valid_losses.get()[start:end], label='OnlyST Valid')
    ax1.set_title("Learning Curve Loss")

    # Plot Recon Loss on the right subplot
    ax2.plot(train_recon_losses.get()[start:end], label='Train')
    ax2.plot(valid_recon_losses.get()[start:end], label='Valid')
    ax2.plot(onlyST_valid_recon_losses.get()[start:end], label='OnlyST Valid')
    ax2.set_title("Learning Curve Recon Loss")

    # Plot Embedding Loss on the left subplot
    ax3.plot(train_embedding_losses.get()[start:end], label='Train')
    ax3.plot(valid_embedding_losses.get()[start:end], label='Valid')
    ax3.plot(onlyST_valid_embedding_losses.get()[start:end], label='OnlyST Valid')
    ax3.set_title("Learning Curve Embedding Loss")

    # Plot Commitment Loss on the right subplot
    ax4.plot(train_commitment_losses.get()[start:end], label='Train')
    ax4.plot(valid_commitment_losses.get()[start:end], label='Valid')
    ax4.plot(onlyST_valid_commitment_losses.get()[start:end], label='OnlyST Valid')
    ax4.set_title("Learning Curve Commitment Loss")

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.tight_layout()
    plt.savefig(save_name)
    return 

def initialize_model(model):
    # init LSTM-attn AE
    for name, param in model.named_parameters():
        if 'weight_ih' in name:  # Weights of the input-hidden for LSTM layers
            nn.init.orthogonal_(param.data)
        elif 'weight_hh' in name:  # Weights of the hidden-hidden (recurrent) for LSTM layers
            nn.init.orthogonal_(param.data)
        elif 'weight' in name:  # Weights of linear layers
            layer_type = name.split('.')[0]
            if isinstance(getattr(model, layer_type), nn.Linear):  # Check if the layer is linear
                nn.init.orthogonal_(param.data)


def run_once(hyper_dir, model_type="ae", condition="b", nameset={"larger": "T", "smaller": "ST"}, noise_controls={"fixlength": False, "amplitude_scale": 0.01}): 
    model_save_dir = os.path.join(hyper_dir, model_type, condition)
    mk(model_save_dir)

    # Loss Recording
    train_losses = ListRecorder(os.path.join(model_save_dir, "train.loss"))
    train_recon_losses = ListRecorder(os.path.join(model_save_dir, "train.recon.loss"))
    train_embedding_losses = ListRecorder(os.path.join(model_save_dir, "train.embedding.loss"))
    train_commitment_losses = ListRecorder(os.path.join(model_save_dir, "train.commitment.loss"))

    valid_losses = ListRecorder(os.path.join(model_save_dir, "valid.loss"))
    valid_recon_losses = ListRecorder(os.path.join(model_save_dir, "valid.recon.loss"))
    valid_embedding_losses = ListRecorder(os.path.join(model_save_dir, "valid.embedding.loss"))
    valid_commitment_losses = ListRecorder(os.path.join(model_save_dir, "valid.commitment.loss"))

    # In C we take onlyST to record the phenomenon-target dataset
    onlyST_valid_losses = ListRecorder(os.path.join(model_save_dir, "valid_onlyST.loss"))
    onlyST_valid_recon_losses = ListRecorder(os.path.join(model_save_dir, "valid_onlyST.recon.loss"))
    onlyST_valid_embedding_losses = ListRecorder(os.path.join(model_save_dir, "valid_onlyST.embedding.loss"))
    onlyST_valid_commitment_losses = ListRecorder(os.path.join(model_save_dir, "valid_onlyST.commitment.loss"))

    text_hist = HistRecorder(os.path.join(model_save_dir, "trainhist.txt"))

    # Recording Directory
    phone_rec_dir = train_cut_phone_
    word_rec_dir = train_cut_word_
    train_guide_path = os.path.join(src_, "guide_train.csv")
    valid_guide_path = os.path.join(src_, "guide_validation.csv")

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


    # Load MV_config
    with open(os.path.join(src_, "mv_config.pkl"), "rb") as file: 
        mv_config = pickle.load(file)

    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type in ["recon4-phi", "recon8-phi", "recon16-phi", "recon32-phi", 
                        "recon48-phi", "recon64-phi", "recon96-phi", "recon128-phi"]: 
        hiddim = int(model_type.split("-")[0].replace("recon", "")) # get hidden dimension from model_type
        # NOTE: such trainings are all on phenomenon dataset and test on that as well, therefore use smaller batch size
        batch_size = 32
        masked_loss = MaskedLoss(loss_fn=nn.MSELoss(reduction="none"))
        # masked_loss = MaskedCosineLoss()    # NOTE: COSINE LOSS! 
        ctc_loss = nn.CTCLoss(blank=mymap.encode("BLANK"))
        model_loss = PseudoAlphaCombineLoss_Recon(masked_loss, ctc_loss, alpha=0.2)
        enc_list = [INPUT_DIM, INTER_DIM_0, INTER_DIM_1, hiddim]
        dec_list = [OUTPUT_DIM, INTER_DIM_0, INTER_DIM_1, hiddim]
        model = AEPPV11(enc_size_list=enc_list, 
                   dec_size_list=dec_list, 
                   ctc_decoder_size_list=ctc_size_list,
                   num_layers=NUM_LAYERS, dropout=DROPOUT, 
                   num_big_layers=NUM_BIG_LAYERS)
    else: 
        raise Exception("Model type not supported! ")
    
    # Load Data
    guide_path = os.path.join(hyper_dir, "guides")
    train_loader = load_data_phenomenon(TestDataset, 
                                    phone_rec_dir, guide_path, load="train", select="both", sampled=False, batch_size=batch_size, 
                                    nameset=nameset, noise_controls=noise_controls, 
                                    mv_config=mv_config)
    valid_loader = load_data_phenomenon(TestDataset, 
                                    phone_rec_dir, guide_path, load="valid", select="both", sampled=True, batch_size=batch_size, 
                                    nameset=nameset, noise_controls=noise_controls, 
                                    mv_config=mv_config)

    model.to(device)
    # initialize_model(model)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    model_str = str(model)
    model_txt_path = os.path.join(model_save_dir, "model.txt")
    with open(model_txt_path, "w") as f:
        f.write(model_str)
        f.write("\n")

    # save pre-train model
    last_model_name = "{}.pt".format(0)
    torch.save(model.state_dict(), os.path.join(model_save_dir, last_model_name))
    num_epochs = 100

    for epoch in range(1, num_epochs + 1):
        text_hist.print("Epoch {}".format(epoch))
        model.train()
        train_loss = 0.
        train_cumulative_l_reconstruct = 0.
        train_cumulative_l_embedding = 0.
        train_cumulative_l_commitment = 0.
        train_num = len(train_loader.dataset)    # train_loader
        for idx, ((x, y_preds), (x_lens, y_preds_lens), pt, sn) in enumerate(train_loader):
            current_batch_size = x.shape[0]
            # y_lens should be the same as x_lens
            optimizer.zero_grad()
            x_mask = generate_mask_from_lengths_mat(x_lens, device=device)
            y_recon = x
            x = x.to(device)
            y_recon = y_recon.to(device)
            y_preds = y_preds.to(device)
            y_preds = y_preds.long()

            (x_hat_recon, x_hat_recon_s), (attn_w_recon, attn_w_preds), (ze, zq) = model(x, x_lens, x_mask)
            # y_hat_preds is now overriden as the list of dec_outs, so no use here. 
            y_hat_preds = x_hat_recon.permute(1, 0, 2)

            l_alpha, (l_reconstruct, l_prediction) = model_loss.get_loss(x_hat_recon, y_recon, 
                                                                         y_hat_preds, y_preds, 
                                                                         x_lens, y_preds_lens, 
                                                                         x_mask)
            
            loss, l_reconstruct, l_embedding, l_commitment = l_alpha, l_reconstruct, l_prediction, l_prediction

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * current_batch_size
            train_cumulative_l_reconstruct += l_reconstruct.item() * current_batch_size
            train_cumulative_l_embedding += l_embedding.item() * current_batch_size
            train_cumulative_l_commitment += l_commitment.item() * current_batch_size

            if idx % 100 == 0:
                text_hist.print(f"""Training step {idx} loss {loss: .3f} \t recon {l_reconstruct: .3f} \t embed {l_embedding: .3f} \t commit {l_commitment: .3f}""")

        train_losses.append(train_loss / train_num)
        train_recon_losses.append(train_cumulative_l_reconstruct / train_num)
        train_embedding_losses.append(train_cumulative_l_embedding / train_num)
        train_commitment_losses.append(train_cumulative_l_commitment / train_num)

        # text_hist.print(f"""※※※Training loss {train_loss / train_num: .3f} \t recon {train_cumulative_l_reconstruct / train_num: .3f} \t embed {train_cumulative_l_embedding / train_num: .3f} \t commit {train_cumulative_l_commitment / train_num: .3f}※※※""")

        last_model_name = "{}.pt".format(epoch)
        torch.save(model.state_dict(), os.path.join(model_save_dir, last_model_name))

        # Valid (ST + T)
        model.eval()
        valid_loss = 0.
        valid_cumulative_l_reconstruct = 0.
        valid_cumulative_l_embedding = 0.
        valid_cumulative_l_commitment = 0.
        valid_num = len(valid_loader.dataset)
        for idx, ((x, y_preds), (x_lens, y_preds_lens), pt, sn) in enumerate(valid_loader):
            current_batch_size = x.shape[0]
            x_mask = generate_mask_from_lengths_mat(x_lens, device=device)

            y_recon = x
            x = x.to(device)
            y_recon = y_recon.to(device)
            y_preds = y_preds.to(device)
            y_preds = y_preds.long()

            (x_hat_recon, x_hat_recon_s), (attn_w_recon, attn_w_preds), (ze, zq) = model(x, x_lens, x_mask)
            y_hat_preds = x_hat_recon.permute(1, 0, 2)

            l_alpha, (l_reconstruct, l_prediction) = model_loss.get_loss(x_hat_recon, y_recon, 
                                                                         y_hat_preds, y_preds, 
                                                                         x_lens, y_preds_lens, 
                                                                         x_mask)
            loss, l_reconstruct, l_embedding, l_commitment = l_alpha, l_reconstruct, l_prediction, l_prediction

            valid_loss += loss.item() * current_batch_size
            valid_cumulative_l_reconstruct += l_reconstruct.item() * current_batch_size
            valid_cumulative_l_embedding += l_embedding.item() * current_batch_size
            valid_cumulative_l_commitment += l_commitment.item() * current_batch_size

        # text_hist.print(f"""※※※Valid loss {valid_loss / valid_num: .3f} \t recon {valid_cumulative_l_reconstruct / valid_num: .3f} \t embed {valid_cumulative_l_embedding / valid_num: .3f} \t commit {valid_cumulative_l_commitment / valid_num: .3f}※※※""")
        valid_losses.append(valid_loss / valid_num)
        valid_recon_losses.append(valid_cumulative_l_reconstruct / valid_num)
        valid_embedding_losses.append(valid_cumulative_l_embedding / valid_num)
        valid_commitment_losses.append(valid_cumulative_l_commitment / valid_num)

        train_losses.save()
        train_recon_losses.save()
        train_embedding_losses.save()
        train_commitment_losses.save()
        valid_losses.save()
        valid_recon_losses.save()
        valid_embedding_losses.save()
        valid_commitment_losses.save()
        onlyST_valid_losses.save()
        onlyST_valid_recon_losses.save()
        onlyST_valid_embedding_losses.save()
        onlyST_valid_commitment_losses.save()


        if epoch % 20 == 0:
            draw_learning_curve_and_accuracy(losses=(train_losses, valid_losses, onlyST_valid_losses), 
                                            recons=(train_recon_losses, valid_recon_losses, onlyST_valid_recon_losses),
                                            embeddings=(train_embedding_losses, valid_embedding_losses, onlyST_valid_embedding_losses),
                                            commitments=(train_commitment_losses, valid_commitment_losses, onlyST_valid_commitment_losses),
                                            start=0, end=epoch,
                                            save_name=os.path.join(model_save_dir, "vis.png"))

    draw_learning_curve_and_accuracy(losses=(train_losses, valid_losses, onlyST_valid_losses), 
                                    recons=(train_recon_losses, valid_recon_losses, onlyST_valid_recon_losses),
                                    embeddings=(train_embedding_losses, valid_embedding_losses, onlyST_valid_embedding_losses),
                                    commitments=(train_commitment_losses, valid_commitment_losses, onlyST_valid_commitment_losses),
                                    start=0, end=epoch,
                                    save_name=os.path.join(model_save_dir, "vis.png"))



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--dataprepare', '-dp', action="store_true")
    parser.add_argument('--timestamp', '-ts', type=str, default="0000000000", help="Timestamp for project, better be generated by bash")
    parser.add_argument('--gpu', '-gpu', type=int, default=0, help="Choose the GPU to work on")
    parser.add_argument('--model','-m',type=str, default = "ae",help="Model type: ae or vqvae")
    parser.add_argument('--condition','-cd',type=str, default="b", help='Condition: b (balanced), u (unbalanced), nt (no-T)')
    parser.add_argument('--noise', '-n', type=float, default=0.01, help="Noise amplitude scale")
    args = parser.parse_args()

    ## Hyper-preparations
    ts = args.timestamp
    train_name = "C_0Tl"
    model_save_dir = os.path.join(model_save_, f"{train_name}-{ts}")
    mk(model_save_dir) 

    if args.dataprepare: 
        # data prepare
        print(f"{train_name}-{ts}-DataPrepare")
        guide_path = os.path.join(model_save_dir, "guides")
        mk(guide_path)
        generate_separation(os.path.join(src_, "phi-T-guide.csv"), 
                            os.path.join(src_, "phi-ST-guide.csv"), 
                            guide_path, 
                            nameset={"larger": "T", "smaller": "ST"})
        
        with open(os.path.join(model_save_dir, "README.note"), "w") as f: 
            f.write("----------------RUN NOTES----------------\n")
            f.write("20241023: Running with AEPPV9, lr=5e-4 and amplitude_scale lower (amplitude=0.004, noise_amplitude=0.0006, f0=50)\n")
            f.write("20241023: sPV/PV, without orthogonal init, with MSE loss, 5-layers\n")
            f.write("20241023: used manual normalization for consistent normalization across the whole dataset. \n")
            f.write("20241023: AEPPV11, multiple encoders and decoders\n")
    else: 
        print(f"{train_name}-{ts}")
        torch.cuda.set_device(args.gpu)
        run_once(model_save_dir, model_type=args.model, condition=args.condition, 
                 nameset={"larger": "T", "smaller": "ST"}, noise_controls={"fixlength": False, "amplitude_scale": 0.004})