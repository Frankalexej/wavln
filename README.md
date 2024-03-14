# Wave Learning (AILab Server Version)

## How to set up enviroment
1. Create directory and its subdirectory "script"
2. Put all files in repo into script
3. use conda env create -f environment.yml to set up environment, if you have anaconda
4. run python paths.py to set up src directories
5. preproc_ prefixed files are used in preprocessing
6. misc_ prefixed files are other utils
7. debug_ prefixed files are for development
8. model_ prefixed files are for modelling
9. test_ prefixed files are for testing stage (not debugging)

## Data
This version uses the LibriSpeech dataset, accompanied by third-party TextGrid annotation files generated with MFA. The TGs contains word-level and phoneme-level annotation. 

## Data Preprocessing
Two tasks: cut the audio according to annotation and organize information into an integrated file. 
- preproc_guide_extract.py: use it if you only want to exract the metadata but not touch the recordings

1. preproc_seg.py: run this file and get the continuous recordings cut into phones or words  
2. preproc_guide_integrate.py: run and integrate the guide files into one large guideline  
3. preproc_guide_mod.py: run and make additional changes to the guide. You can self-define any change because this is post-hoc. Currently it includes "destress" (disregarding the stress markings), "addpath" (add extra path combined from rec and idx. This will take around twice the size in storage but will save calculation time when loading dataset), "markspeaker" (separately mark the speaker of the recording, this can help with later separation of training, validation and testing sets)  
4. preproc_guide_sepTVT.py: separate training, validation as well as test dataset. This will make sure any speaker (not only segments) is only in one of the sets. 



## Frank's Notes
1. The phonetic aligment (&transcription) is using [ARPABET](https://en.wikipedia.org/wiki/ARPABET), with alphabet (combination)s marking sounds and numbers noting stress. 

2. We can just leave the structure of the dataset as it is after cutting. Since it will just change the path of files, it won't really affect the reading efficiency during training.

3. 20240205 todo: try bidirectional LSTM for prediction CTC, this would require us to check how to deal with padding. If that does not work either, check what Mockingjoy is doing and try to replicate that with LSTM. 



Frank Modification Note (since 20240216)
<!-- - 20240216 1: Instead of trying to limit the hidden representation, we first try to restore the hierarchical multiscale LSTM. This is because we gave up the model during the time when we trained the model on the wrong data. This time, since we are on the corrected data, I want to try this and check whether it will learn the correct thing.  -->
- 20240216 2: We will try VQ-VAE first. To achieve this, let's try VAE first. 
- 20240217 1: Today we tried VAE. The result was not satisfactory. Tried to remove the V part and the result was similar as well. It is said that the KL divergence can control the disentaglement of hidden representation, and probably it will benefit our learning goal. So tomorrow I will try to increase the weight of KL divergence and check the result. In case it does not work, we will proceed to check whether VQ will work. Somehow, though, the VQ seems to be based on VAE. Yet since we tried VAE and non-VAE with same structure, we are now pretty sure that, despite better reconstruction quality for non-VAE, VAE is not so bad in terms of hidden representation learning (it is similar). We can try VQVAE first, if not working, we can also try VQAE.
- 20240218 1: Today we tried beta-VAE, and tried beta=5 and 1.5. Neither worked better than beta=1. This means that the problem does not lie here. For the next step I am thinking about VQVAE. It seems that VAEs usually work with CNN and non-sequential data; however, we cannot do it that way. This is because CNN does not seem very suitable for getting **a sequence of hidden representations**, which we need in essence. Considering Begus' works, they don't really target sequences longer than, at most, a small sequence of phonemes, which look similar to ours but their model treat the small sequence as a whole. So what we want is different. So for the next step we either test VQ-VAE or adapt this model to VQAE (then similar to SE19). 
- 20240219 1: Today we tried VQVAE and tested it with our old testing methods + downstream task [i.e. phoneme prediction]. Now it seems that it does not help to any degree, instead, it makes the situation even worse. So for the next step I don't really know how to step forward, one idea I have in mind is to check boundary prediction, but this may not return satisfactory results. Another one is to check CNN, but we will be dealing with the problem of how to deal with sequential hidden representation. 
- 20240314: For the next step we want to see whether old old training model (i.e. normal attention AE model trained on full set) is able to grab the contextual contrast as well. This idea is inspired by the observation that the model trained on deaspiration data showed almost no learning of phonemic contrast, while having quite well learned the allophonic contrast bewteen aspirated and unaspirated plosives. Because our full-set trained model has almost no learning at all about phonemic knowledge, we suspect that the model is in fact learning from contrast. And it could be confirmed if the full-set model is also good at grabing the allophonic contrast. We suspect that the reason behidne this is that during the training we provided enough contextual information, which enables the mdoel to better generalize and categories the differernt tokens according to contrast, while 0since there are no lexical taggings combined with words (no lexical contrast), it would appear to the mdoel that the different phonemes are not very different. 
- Therefore, as the next step after trying with the full-set model, we need to test with lexical contrast, whcih will tell us whether the model is really utilizing this additional contrast to do categorization. Just like humans. 