#!/bin/bash
#SBATCH --job-name=multi30k_translation_%j
#SBATCH --output=logs/multi30k_translation_%j.out
#SBATCH --error=logs/multi30k_translation_%j.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=driggersellis.cw@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000mb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --time=12:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

module load mamba openssl

mamba activate optical-env

echo "Source Lang = $(echo $src)"
echo "Target Lang = $(echo $tgt)"
echo "Dataset = $(echo $dataset)"

T1=$(date +%s)
python multi30k-scripts/translate_multi30k.py -s $src -t $tgt -d $dataset
T2=$(date +%s)

mamba deactivate

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
