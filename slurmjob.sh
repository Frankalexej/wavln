#!/bin/bash
#SBATCH --job-name=sandbox_test
#SBATCH --output=/mnt/storage/franklhtan/wavln/test_job.out
#SBATCH --error=/mnt/storage/franklhtan/wavln/test_job.err
#SBATCH --time=00:10:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --chdir="/mnt/franklhtan/projects/wavln/scripts/"

singularity exec --nv --fakeroot --writable /home/franklhtan/ubuntu \
bash -c "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate wavln && python test.py"
