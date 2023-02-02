#!/bin/bash
#SBATCH --time=5:10:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -o 'logs/%A_%a.log'

set -e

if [[ "$HOSTNAME" == *"tiger"* ]]
then
    module load anaconda
    conda activate 247-main
elif [[ "$HOSTNAME" == *"della"* ]]
then
    echo "It's Della"
    # module load anaconda3/2021.11
    module load anaconda
    module load cudatoolkit/12.0
    module load cudnn/cuda-11.x/8.2.0
    source activate /home/hgazula/.conda/envs/247-main-tf
    # source activate 247-main
else
    module load anacondapy
    source activate 247-main
fi

echo 'Requester:' $USER
echo 'Node:' $HOSTNAME
echo "$@"
for run in $(seq 1 6); do
    echo 'Run start time:' `date`
    python "$@"
    echo 'Run end time:' `date`
done

echo 'Ensemble start:' `date`
python "$@" --ensemble
echo 'End time:' `date`
