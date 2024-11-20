#!/bin/bash

# Function to generate a 10-digit random number
generate_random_number() {
    number=""
    for i in {1..10}; do
        digit=$((RANDOM % 10))
        number="${number}${digit}"
    done
    echo "$number"
}

# Arrays of options for each argument
ms=('recon4-phi' 'recon8-phi' 'recon16-phi' 'recon32-phi' 'recon48-phi' 'recon64-phi') # 'recon4-phi' 'recon8-phi' 'recon16-phi' 'recon32-phi' 'recon48-phi' 'recon64-phi' 'recon96-phi'
# ms=('recon32-phi')
cs=('b') # 
# zls=('hidrep' 'attnout' 'ori' 'enc-lin1' 'dec-lin1' 'enc-rnn1-f' 'enc-rnn1-b' 'dec-rnn1-f' 'enc-rnn2-f' 'enc-rnn2-b' 'dec-rnn2-f' 'enc-rnn3-f' 'enc-rnn3-b' 'dec-rnn3-f' 'enc-rnn4-f' 'enc-rnn4-b' 'dec-rnn4-f' 'enc-rnn5-f' 'enc-rnn5-b' 'dec-rnn5-f')
# zls=('ori')
# zls=('POS' 'VC' 'STOPPOA' 'PARC-0-2' 'PARC-2-4' 'PARC-4-6' 'PPP' 'PPH' 'GENDER' 'ARC-0-2' 'ARC-2-4' 'ARC-4-6' 'ARC-6-8' 'ARC-8-99')    # 'POS' 'VC' 'STOPPOA' 'PARC-0-2' 'PARC-2-4' 'PARC-4-6' 'PPP' 'PPH' 'GENDER'
zls=('AAIY')
# zls=('AACrossPos')
# zls=('ARC-0-2' 'ARC-2-4' 'ARC-4-6' 'ARC-6-8' 'ARC-8-99') 
# 'hidrep' 'attnout' 'ori' 'enc-lin1' 'dec-lin1' 'enc-rnn1-f' 'enc-rnn1-b' 'dec-rnn1-f' 'enc-rnn2-f' 'enc-rnn2-b' 'dec-rnn2-f' 'enc-rnn3-f' 'enc-rnn3-b' 'dec-rnn3-f' 'enc-rnn4-f' 'enc-rnn4-b' 'dec-rnn4-f' 'enc-rnn5-f' 'enc-rnn5-b' 'dec-rnn5-f'
ts='1113024340'
tn="ABXSomethingAll-first-data-AAIY" # FinalEpochsDimneutral FinalEpochs ABXSomething-vowel-vowel-speaker
echo "Timestamp: $ts; Test: $tn"

# Loop from 1 to 10, incrementing by 1
# Loop over each combination of arguments
for m in "${ms[@]}"; do
    for c in "${cs[@]}"; do
        for zl in "${zls[@]}"; do
            # Randomly select a GPU between 0 and 8
            gpu=$((RANDOM % 9))
            # Run the Python script with the current combination of arguments in the background
            python E_0A_e_tests.py -ts "$ts" -m "$m" -cd "$c" -gpu "$gpu" -zl "$zl" -tn "$tn"&
        done
    done
done
# Wait for all background processes to finish
wait
