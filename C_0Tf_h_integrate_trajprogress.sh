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
ms=('recon4-phi' 'recon8-phi' 'recon16-phi' 'recon32-phi' 'recon48-phi' 'recon64-phi' 'recon96-phi')   # 'recon4-phi' 'recon8-phi' 'recon16-phi' 'recon32-phi' 'recon48-phi' 'recon64-phi' 'recon96-phi' 'recon128-phi'
cs=('b')

# Generate a 10-digit random number
# ts='0826140610'
# ts='0902011400'
# ts='0904231859'
ts='0910145009'
echo "Timestamp: $ts"

# Loop from 1 to 10, incrementing by 1
# Loop over each combination of arguments
for m in "${ms[@]}"; do
    for c in "${cs[@]}"; do
        # Randomly select a GPU between 0 and 8
        gpu=$((RANDOM % 9))
        # Run the Python script with the current combination of arguments in the background
        python C_0Tf_h_integrate_trajprogress.py -ts "$ts" -m "$m" -cd "$c" -gpu "$gpu" &
    done
done
# Wait for all background processes to finish
wait
