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
ms=('recon16-phi') # 'recon4-phi' 'recon8-phi' 'recon16-phi' 'recon32-phi' 'recon48-phi' 'recon64-phi' 'recon96-phi'
cs=('b')
is=(1) # 
# Generate a 10-digit random number
# ts='0910145009'     # cosine loss trained, default init, AEPPV9, lr=1e-4, noise=0.004
ts='1014180320'     # cosine loss trained, default init, AEPPV9, lr=5e-4, noise=0.004
echo "Timestamp: $ts"

# Loop from 1 to 10, incrementing by 1
# for (( i=1; i<=5; i++ )); do
for i in "${is[@]}"; do
    # Loop over each combination of arguments
    for m in "${ms[@]}"; do
        for c in "${cs[@]}"; do
            # Randomly select a GPU between 0 and 8
            gpu=$((RANDOM % 9))
            # Run the Python script with the current combination of arguments in the background
            python C_0Ti_b_eval.py -ts "$ts" -rn "$i" -m "$m" -cd "$c" -gpu "$gpu" &
        done
    done
done

# Wait for all background processes to finish
wait
