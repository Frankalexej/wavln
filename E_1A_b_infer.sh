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
is=(4) # 
# Generate a 10-digit random number
# ts='0910145009'     # cosine loss trained, default init, AEPPV9, lr=1e-4, noise=0.004
ts='1124190101'     # cosine loss trained, default init, AEPPV9, lr=5e-4, noise=0.004
echo "Timestamp: $ts"

# # Loop from 1 to 10, incrementing by 1
# # for (( i=1; i<=5; i++ )); do
# for i in "${is[@]}"; do
#     # Loop over each combination of arguments
#     for m in "${ms[@]}"; do
#         for c in "${cs[@]}"; do
#             # Randomly select a GPU between 0 and 8
#             gpu=$((RANDOM % 9))
#             # Run the Python script with the current combination of arguments in the background
#             python E_1A_b_infer.py -ts "$ts" -rn "$i" -m "$m" -cd "$c" -gpu "$gpu" &
#         done
#     done
# done

# # Wait for all background processes to finish
# wait


# Fix Runs
tasks=(
    # "4 1 b"
    # "4 2 b"
    # "4 3 b"
    # "4 4 b"
    # "4 5 b"
    # "8 1 b"
    # "8 2 b"
    # "8 3 b"
    # "8 4 b"
    # "8 5 b"
    # "16 1 b"
    # "16 2 b"
    # "16 3 b"
    # "16 5 b"
    # "32 1 b"
    # "32 2 b"
    # "32 3 b"
    # "32 4 b"
    # "32 5 b"
    # "48 1 b"
    # "48 2 b"
    # "48 3 b"
    "48 4 b"
    # "48 5 b"
    # "64 1 b"
    # "64 2 b"
    # "64 3 b"
    "64 4 b"
    # "64 5 b"
)

# Iterate over the tasks 

for task in "${tasks[@]}"; do 
    # Read the loop values 
    read mnum i c <<< "$task" 
    echo "Running recon$mnum-phi with $i runs and $c condition" 
    # Replace the following line with your actual command 
    gpu=$((RANDOM % 9))
    python E_1A_b_infer.py -ts "$ts" -rn "$i" -m "recon$mnum-phi" -cd "$c" -gpu "$gpu" &

done 
# Wait for all background processes to finish
wait