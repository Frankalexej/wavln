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
ms=('recon4-phi' 'recon8-phi' 'recon16-phi' 'recon32-phi' 'recon48-phi' 'recon64-phi' 'recon96-phi') # 'recon4-phi' 'recon8-phi' 'recon16-phi' 'recon32-phi' 'recon48-phi' 'recon64-phi' 'recon96-phi' 'recon128-phi'
cs=('b') # 
zls=('reconori') 
# 'hidrep' 'attnout' 'ori' 'enc-lin1' 'dec-lin1' 'enc-rnn1-f' 'enc-rnn1-b' 'dec-rnn1-f' 'enc-rnn2-f' 'enc-rnn2-b' 'dec-rnn2-f'
# Generate a 10-digit random number
# ts='0821210446'
# ts='0904231859'
ts='0912014617'
tn="reconloss-stop-stop"
echo "Timestamp: $ts; Test: $tn"

# Loop from 1 to 10, incrementing by 1
# Loop over each combination of arguments
for m in "${ms[@]}"; do
    for c in "${cs[@]}"; do
        for zl in "${zls[@]}"; do
            # Randomly select a GPU between 0 and 8
            gpu=$((RANDOM % 9))
            # Run the Python script with the current combination of arguments in the background
            python C_0Tf_o_well_formedness.py -ts "$ts" -m "$m" -cd "$c" -gpu "$gpu" -zl "$zl" -tn "$tn"&
        done
    done
done
# Wait for all background processes to finish
wait
