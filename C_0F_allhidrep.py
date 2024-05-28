import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
from model_dataset import TokenMap
from model_model import AEPPV1, AEPPV2

from model_dataset import WordDatasetPath as ThisDataset
from C_0X_defs import *
# from C_0D_run import load_data_general

def load_data_general(dataset, rec_dir, target_path, load="train", select=0.3, sampled=True, batch_size=BATCH_SIZE):
    # for general, path is easy, let's just load it
    integrated = pd.read_csv(target_path)
    # integrated = integrated.sample(frac=1).reset_index(drop=True)

    mytrans = TheTransform(sample_rate=REC_SAMPLE_RATE, 
                        n_fft=N_FFT, n_mels=N_MELS, 
                        normalizer=Normalizer.norm_mvn, 
                        denormalizer=DeNormalizer.norm_mvn)
    
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
                        transform=mytrans)
    
    use_len = int(select * len(ds))
    remain_len = len(ds) - use_len
    use_ds, remain_ds = random_split(ds, [use_len, remain_len])

    use_shuffle = True if load == "train" else False
    loader = DataLoader(use_ds, batch_size=batch_size, shuffle=use_shuffle, num_workers=LOADER_WORKER, collate_fn=dataset.collate_fn)
    return loader

def get_included_names(loader):
    included_names = []
    for (x, x_lens, name) in loader: 
        name = name[0]
        included_names += [name]
    return included_names

######################## Full Hidrep Evaluation ########################
# full_hidrep_evaluate_list = [
#     ["AA", "UW"], 
#     ["AA", "IY"],

# ]

def cutHid(hid, cutstart, cutend, start_offset=0, end_offset=1): 
    selstart = max(cutstart, int(cutstart + (cutend - cutstart) * start_offset))
    selend = min(cutend, int(cutstart + (cutend - cutstart) * end_offset))
    # hid is (L, H)
    return hid[selstart:selend, :]

def get_toplot(data, name_dict, df, selector, max_counts=500, offsets=(0, 1)): 
    selected_df = pd.DataFrame(columns=df.columns)
    for item in selector:
        # Filter the DataFrame for the current item
        filtered_df = df[df['segment_nostress'] == item]
        
        # Check if the number of rows exceeds the maximum count
        if len(filtered_df) > max_counts:
            # Randomly sample max_counts[item] rows
            sampled_df = filtered_df.sample(n=max_counts, replace=False)
        else:
            sampled_df = filtered_df
        # Append to the selected_df
        selected_df = pd.concat([selected_df, sampled_df], axis=0)
    # selected_df = df[df["segment_nostress"].isin(cluster_groups)]
    selected_wuid = selected_df["wuid"].tolist()
    indices = [name_dict[token] for token in selected_wuid]
    selected_items = []
    for idx in indices: 
        selected_items.append(data[idx])
    cutstarts = selected_df["startFrame"]
    cutends = selected_df["endFrame"]

    hid_sel = np.empty((0, 8))
    tag_sel = []
    for (item, start, end, tag) in zip(selected_items, cutstarts, cutends, selected_df["segment_nostress"]): 
        hid = cutHid(item, start, end, offsets[0], offsets[1])
        hidlen = hid.shape[0]
        hid_sel = np.concatenate((hid_sel, hid), axis=0)
        tag_sel += [tag] * hidlen
    return hid_sel, np.array(tag_sel)
######################## Full Hidrep Evaluation End of Definition ########################

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
                                 file_prefix=f"fullhid-{stop_epoch}")
    all_ze = []
    all_zq = []
    all_name = []

    for (x, x_lens, name) in single_loader: 
        name = name[0]

        x_mask = generate_mask_from_lengths_mat(x_lens, device=device)
        
        x = x.to(device)

        ze, zq = model.encode(x, x_lens, x_mask)

        ze = ze.cpu().detach().numpy().squeeze()
        zq = zq.cpu().detach().numpy().squeeze()

        # all_ze += [ze]
        all_zq += [zq]
        all_name += [name]
    
    # reshandler.res["ze"] = all_ze
    reshandler.res["zq"] = all_zq
    reshandler.res["name"] = all_name

    reshandler.save()
    print(f"Results fullhid-{stop_epoch} saved at {res_save_dir}")
    return 0

def evaluate_hidrep_one_epoch(hidreps, guide_file, name_dict, evaluation_pairs, example_save_path, phoneme_map): 
    sil_scores = []
    example_pair = evaluation_pairs[0]
    X, Y = get_toplot(data=hidreps, 
                    name_dict=name_dict, 
                    df=guide_file, 
                    selector=example_pair, 
                    max_counts=500, 
                    offsets=(0.4, 0.6))
    sil_scores.append(silhouette_score(X, Y))
    pca = PCA(n_components=2)  # Reduce to 2 dimensions
    pca_result = pca.fit_transform(X)  # Make sure to convert from PyTorch tensor to NumPy array if necessary
    plt.figure(figsize=(10, 7))  # Larger figure size for better visibility
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=np.vectorize(phoneme_map.encode)(Y), cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Hidden Representations between {} and {}'.format(*example_pair))
    plt.savefig(example_save_path)
    plt.close()
    
    for pair in evaluation_pairs[1:]: 
        X, Y = get_toplot(data=hidreps, 
                            name_dict=name_dict, 
                            df=guide_file, 
                            selector=pair, 
                            max_counts=500, 
                            offsets=(0.4, 0.6))
        sil_scores.append(silhouette_score(X, Y))
    return sil_scores



def main(train_name, ts, run_number, model_type, model_save_dir, res_save_dir, guide_dir, word_guide_, run_eval="re"): 
    # Dirs
    rec_dir = train_cut_phone_
    # Check model path
    assert PU.path_exist(model_save_dir)
    assert PU.path_exist(guide_dir)

    # Load data
    word_rec_dir = train_cut_word_
    valid_guide_path = os.path.join(src_, "guide_validation.csv")
    single_loader = load_data_general(ThisDataset, 
                                      word_rec_dir, valid_guide_path, load="valid", select=0.3, sampled=False, 
                                      batch_size=1)

    with open(os.path.join(src_, "no-stress-seg.dict"), "rb") as file:
        # Load the object from the file
        mylist = pickle.load(file)
        mylist = ["BLANK"] + mylist
        mylist = mylist + ["SIL"]   # this is to fit STV vs #TV

    # Now you can use the loaded object
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
    else: 
        raise Exception("Model type not supported! ")

    if run_eval == "re" or run_eval == "r":
        # Run model and save results        
        for epoch in range(0, 100): 
            run_one_epoch(model, single_loader, None, model_save_dir, epoch, res_save_dir)

    # Prepare guide file for hidrep evaluation        
    included_names = get_included_names(single_loader)
    # read in guide file
    guide_file = pd.read_csv(valid_guide_path)
    # filtering out is not necessary, since we only include wuid for encoded words
    guide_file = guide_file[~guide_file["segment_nostress"].isin(["sil", "sp", "spn"])]
    filtered_df = guide_file[guide_file['wuid'].isin(included_names)].copy()
    filtered_df["startFrame"] = filtered_df.apply(lambda x: time_to_frame(x['startTime'] - x['word_startTime']), axis=1)
    filtered_df["endFrame"] = filtered_df.apply(lambda x: time_to_frame(x['endTime'] - x['word_startTime']), axis=1)
    name_dict = {token: index for index, token in enumerate(included_names)}

    all_v = filtered_df[filtered_df["segment_nostress"].isin(ARPABET.list_vowels())]["segment_nostress"].unique().tolist()
    full_hidrep_evaluate_list = list(combinations(all_v, 2))

    if run_eval == "re" or run_eval == "e":
        # Evaluate hidrep
        cross_epoch_sil_scores = []
        for epoch in range(0, 100): 
            reshandler = DictResHandler(whole_res_dir=res_save_dir, 
                                        file_prefix=f"fullhid-{epoch}")
            reshandler.read()
            hidreps = reshandler.res["zq"]
            example_save_path = os.path.join(res_save_dir, f"fullhidex-{epoch}.png")
            sil_scores = evaluate_hidrep_one_epoch(hidreps, filtered_df, name_dict, full_hidrep_evaluate_list, example_save_path, mymap)

            sil_score_path = os.path.join(res_save_dir, f"sil_scores_{epoch}.pk")
            with open(sil_score_path, "wb") as file: 
                pickle.dump(sil_scores, file)
            cross_epoch_sil_scores.append(np.array(sil_scores).mean())
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--timestamp', '-ts', type=str, default="0000000000", help="Timestamp for project, better be generated by bash")
    parser.add_argument('--runnumber', '-rn', type=str, default="0", help="Run number")
    parser.add_argument('--gpu', '-gpu', type=int, default=0, help="Choose the GPU to work on")
    parser.add_argument('--model','-m',type=str, default = "ae",help="Model type: ae or vqvae")
    parser.add_argument('--condition','-cd',type=str, default="b", help='Condition: b (balanced), u (unbalanced), nt (no-T)')
    parser.add_argument('--runeval','-re',type=str, default="re", help='re, r or e')
    args = parser.parse_args()

    # set device number
    torch.cuda.set_device(args.gpu)

    ts = args.timestamp # this timestamp does not contain run number
    rn = args.runnumber
    runeval = args.runeval
    train_name = "C_0F"
    if not PU.path_exist(os.path.join(model_save_, f"{train_name}-{ts}-{rn}")):
        raise Exception(f"Training {train_name}-{ts}-{rn} does not exist! ")
    
    model_save_dir = os.path.join(model_save_, f"{train_name}-{ts}-{rn}", args.model, args.condition)
    guide_dir = os.path.join(model_save_, f"{train_name}-{ts}-{rn}", "guides")
    res_save_dir = os.path.join(model_save_, f"eval-{train_name}-{ts}")
    this_model_condition_dir = os.path.join(res_save_dir, args.model, args.condition, f"{rn}")
    valid_full_guide_path = os.path.join(src_, "guide_validation.csv")
    mk(this_model_condition_dir)

    main(train_name, ts, args.runnumber, args.model, model_save_dir, this_model_condition_dir, guide_dir, valid_full_guide_path, runeval)