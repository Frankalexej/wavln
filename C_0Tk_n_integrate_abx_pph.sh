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
cs=('b') # 
# zls=('hidrep' 'attnout' 'ori' 'enc-lin1' 'dec-lin1' 'enc-rnn1-f' 'enc-rnn1-b' 'dec-rnn1-f' 'enc-rnn2-f' 'enc-rnn2-b' 'dec-rnn2-f' 'enc-rnn3-f' 'enc-rnn3-b' 'dec-rnn3-f' 'enc-rnn4-f' 'enc-rnn4-b' 'dec-rnn4-f' 'enc-rnn5-f' 'enc-rnn5-b' 'dec-rnn5-f')
zls=('hidrep' 'attnout' 'ori')
# zls=('ARC-0-2' 'ARC-2-4' 'ARC-4-6' 'ARC-6-8')     # 'POS' 'PPP' 'PPH' 'VC'
# 'hidrep' 'attnout' 'ori' 'enc-lin1' 'dec-lin1' 'enc-rnn1-f' 'enc-rnn1-b' 'dec-rnn1-f' 'enc-rnn2-f' 'enc-rnn2-b' 'dec-rnn2-f' 'enc-rnn3-f' 'enc-rnn3-b' 'dec-rnn3-f' 'enc-rnn4-f' 'enc-rnn4-b' 'dec-rnn4-f' 'enc-rnn5-f' 'enc-rnn5-b' 'dec-rnn5-f'
ts='1021183234'
tn="ABXSomethingAll-stop-data-aspirationRangeComp-2-4" # FinalEpochsDimneutral ABXpositionAll-vowel-vowel-vowel
echo "Timestamp: $ts; Test: $tn"
# portionrange=("0 2" "2 4" "4 6" "6 8" "8 99")

# Loop from 1 to 10, incrementing by 1
# Loop over each combination of arguments
# for pr in "${portionrange[@]}"; do
#     # Split the pair into two variables
#     set -- $pr
#     start_range=$1
#     end_range=$2
#     tn_appended="${tn}-${start_range}-${end_range}"
#     echo "Portion range: $pr"
#     # Array to hold process IDs of background processes
#     pids=()

# for m in "${ms[@]}"; do
#     for c in "${cs[@]}"; do
#         for zl in "${zls[@]}"; do
#             # Randomly select a GPU between 0 and 8
#             gpu=$((RANDOM % 9))
#             # Run the Python script with the current combination of arguments in the background
#             python C_0Tk_n_integrate_abx_pph.py -ts "$ts" -m "$m" -cd "$c" -gpu "$gpu" -zl "$zl" -tn "$tn"&
#             pids+=($!)  # Store the PID of the background process
#         done
#     done
# done
    # Wait for all background processes to complete
    # for pid in "${pids[@]}"; do
    #     wait "$pid"
    # done
# done

for m in "${ms[@]}"; do
    for c in "${cs[@]}"; do
        for zl in "${zls[@]}"; do
            # Randomly select a GPU between 0 and 8
            gpu=$((RANDOM % 9))
            # Run the Python script with the current combination of arguments in the background
            python C_0Tk_n_integrate_abx_pph.py -ts "$ts" -m "$m" -cd "$c" -gpu "$gpu" -zl "$zl" -tn "$tn"&
        done
    done
done
# Wait for all background processes to finish
wait
