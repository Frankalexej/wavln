#!/bin/bash

# ====== CONFIGURABLE PART ======
ts='0912014617'   # your timestamp
cs=('b')          # your CDs
gpu=0             # fixed GPU

# Combined list: each element is "model_run"
ms_is=(
        'recon4-phi_1' 
        'recon4-phi_2'
        'recon4-phi_3'
        'recon4-phi_4'
        'recon4-phi_5'
        'recon8-phi_1'
        'recon8-phi_2'
        'recon8-phi_3'
        'recon8-phi_4'
        'recon8-phi_5'
        'recon16-phi_1'
        'recon16-phi_2'
        'recon16-phi_3'
        'recon16-phi_4'
        'recon16-phi_5'
        'recon32-phi_1'
        'recon32-phi_2'
        'recon32-phi_3'
        'recon32-phi_4'
        'recon32-phi_5'
        )

# Max number of parallel jobs
MAX_CONCURRENT=3
# =================================

echo "Timestamp: $ts"
running_jobs=0

for entry in "${ms_is[@]}"; do
    # Split into model and run number
    m="${entry%%_*}"    # part before underscore
    i="${entry##*_}"    # part after underscore

    for c in "${cs[@]}"; do
        echo "Launching: m=$m, i=$i, c=$c"
        python C_0Tf_ba_eval_fakeswapped.py -ts "$ts" -rn "$i" -m "$m" -cd "$c" -gpu "$gpu" &

        ((running_jobs++))

        if (( running_jobs >= MAX_CONCURRENT )); then
            wait -n
            ((running_jobs--))
        fi
    done
done

wait  # Wait for any remaining jobs to finish
