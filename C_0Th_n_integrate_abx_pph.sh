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
ms=('recon4-phi' 'recon4-phi' 'recon8-phi' 'recon16-phi' 'recon32-phi' 'recon48-phi' 'recon64-phi' 'recon96-phi') # 'recon4-phi' 'recon8-phi' 'recon16-phi' 'recon32-phi' 'recon48-phi' 'recon64-phi' 'recon96-phi'
cs=('b') # 
zls=('hidrep' 'attnout' 'enc-lin1' 'dec-lin1' 'enc-rnn1-l' 'enc-rnn1-f' 'enc-rnn1-b' 'dec-rnn1-f' 'enc-rnn2-l' 'enc-rnn2-f' 'enc-rnn2-b' 'dec-rnn2-f' 'enc-rnn3-l' 'enc-rnn3-f' 'enc-rnn3-b' 'dec-rnn3-f' 'enc-rnn4-l' 'enc-rnn4-f' 'enc-rnn4-b' 'dec-rnn4-f' 'enc-rnn5-f' 'enc-rnn5-b' 'dec-rnn5-f') 
# 'hidrep' 'attnout' 'ori' 'enc-lin1' 'dec-lin1' 'enc-rnn1-l' 'enc-rnn1-f' 'enc-rnn1-b' 'dec-rnn1-f' 'enc-rnn2-l' 'enc-rnn2-f' 'enc-rnn2-b' 'dec-rnn2-f' 'enc-rnn3-l' 'enc-rnn3-f' 'enc-rnn3-b' 'dec-rnn3-f' 'enc-rnn4-l' 'enc-rnn4-f' 'enc-rnn4-b' 'dec-rnn4-f' 'enc-rnn5-f' 'enc-rnn5-b' 'dec-rnn5-f'
# Generate a 10-digit random number
# ts='0821210446'
# ts='0904231859'
ts='0919162922'
tn="clusterARI-vowel-vowel"
echo "Timestamp: $ts; Test: $tn"

# Loop from 1 to 10, incrementing by 1
# Loop over each combination of arguments
for m in "${ms[@]}"; do
    for c in "${cs[@]}"; do
        for zl in "${zls[@]}"; do
            # Randomly select a GPU between 0 and 8
            gpu=$((RANDOM % 9))
            # Run the Python script with the current combination of arguments in the background
            python C_0Th_n_integrate_abx_pph.py -ts "$ts" -m "$m" -cd "$c" -gpu "$gpu" -zl "$zl" -tn "$tn"&
        done
    done
done
# Wait for all background processes to finish
wait
