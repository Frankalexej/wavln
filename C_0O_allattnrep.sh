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
# ms=('mtl')
ms=('recon8-5l')
runs=("1")
cs=('u')
eps=(0 5 10 15 20 25 30 35 40 45)
# 10 20 30 40
ep_jump=5

# Generate a 10-digit random number
# ts='0526172101'
ts='0609021600'
echo "Timestamp: $ts"
re='a'
echo "Runeval: $re"

for m in "${ms[@]}"; do
    for r in "${runs[@]}"; do
        for c in "${cs[@]}"; do
            for eps in "${eps[@]}"; do
                # Randomly select a GPU between 0 and 8
                gpu=$((RANDOM % 9))
                # Run the Python script with the current combination of arguments in the background
                python C_0O_allattnrep.py -ts "$ts" -rn "$r" -m "$m" -cd "$c" -gpu "$gpu" -re "$re" -eps "$eps" -epe "$(($eps + $ep_jump))" &
            done
        done
    done
done

# # Loop from 1 to 10, incrementing by 1
# for (( i=1; i<=5; i++ )); do
#     # Loop over each combination of arguments
#     for m in "${ms[@]}"; do
#         for c in "${cs[@]}"; do
#             # Randomly select a GPU between 0 and 8
#             gpu=$((RANDOM % 9))
#             # Run the Python script with the current combination of arguments in the background
#             python C_0D_eval.py -ts "$ts" -rn "$i" -m "$m" -cd "$c" -gpu "$gpu" &
#         done
#     done
# done

# Wait for all background processes to finish
wait
