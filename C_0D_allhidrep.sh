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
ms=('ae' 'aefixemb')
runs=("1 4 5" "1 3 4 5")
cs=('u')

# Generate a 10-digit random number
ts='0412203108'
echo "Timestamp: $ts"

for m in "${!ms[@]}"; do
    for r in ${runs[m]}; do
        for c in "${cs[@]}"; do
            # Randomly select a GPU between 0 and 8
            gpu=$((RANDOM % 9))
            # Run the Python script with the current combination of arguments in the background
            python C_0D_allhidrep.py -ts "$ts" -rn "$r" -m "${ms[m]}" -cd "$c" -gpu "$gpu" &
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
