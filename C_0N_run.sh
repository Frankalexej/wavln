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
ms=('reconcus-4-3' 'reconcus-6-3' 'reconcus-8-3' 'reconcus-10-3' 'reconcus-12-3' 'reconcus-16-3' 'reconcus-20-3')
cs=('u')

# Generate a 10-digit random number
ts=$(date +"%m%d%H%M%S")
# ts='0527172800'
echo "Timestamp: $ts"

# Loop from 1 to 10, incrementing by 1
for (( i=1; i<=1; i++ )); do
    # Loop over each combination of arguments
    python C_0N_run.py -ts "$ts-$i" -dp
    for m in "${ms[@]}"; do
        for c in "${cs[@]}"; do
            # Randomly select a GPU between 0 and 8
            gpu=$((RANDOM % 9))
            # Run the Python script with the current combination of arguments in the background
            python C_0N_run.py -ts "$ts-$i" -m "$m" -cd "$c" -gpu "$gpu" &
        done
    done
done

# Wait for all background processes to finish
wait
