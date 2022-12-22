#!/bin/bash
#SBATCH --time=20:10:00
#SBATCH --mem=32GB
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
    module load anaconda
    source activate 247-main
else
    module load anacondapy
    source activate srm
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
