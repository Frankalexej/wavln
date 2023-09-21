# Wave Learning





## Files
### dataset.py
dataset.py intends to include all datasets in it and unify the interface of custom dataset for this project. 








# Thinkings
1. We will mainly play with sound and see the differences in hidden representation. For out case there is no simple way to transfer spectrogram back to sound without loss. It is therefore more feasible to do the more natural ones. 

2. [20230921]:   
    a. check phoneme frame count, see whether it is possible to have these single phonemes, instead of having to adjust the model's mel window size (now at least 400 frames)   
    -> 29066 / 862459 <= 400 frames
    b. if okay, check whether it can be easily fed into existing dataset.    
    c. and work   