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
ms=('ae' 'vqvae')
cs=('u')

# Generate a 10-digit random number
ts='0308012232'
echo "Timestamp: $ts"

# Loop from 1 to 10, incrementing by 1
# Loop over each combination of arguments
for c in "${cs[@]}"; do
    # Randomly select a GPU between 0 and 8
    gpu=$((RANDOM % 9))
    # Run the Python script with the current combination of arguments in the background
    python C_0A_integrate_eval.py -ts "$ts" -cd "$c" -gpu "$gpu" &
done

# Wait for all background processes to finish
wait
