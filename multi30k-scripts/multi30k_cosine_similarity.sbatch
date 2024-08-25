#!/bin/bash
#SBATCH --job-name=multi30k_cosine_similarity_%j
#SBATCH --output=logs/multi30k_similarity_%j.out
#SBATCH --error=logs/multi30k_similarity_%j.err
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

mamba activate multi30k-env

echo "Cosine Similarity Calculation for Multi30k Translation"
echo "Source Lang = $(echo $src)"
echo "Target Lang = $(echo $tgt)"

T1=$(date +%s)
python multi30k-scripts/cosine_similarity_multi30k.py -s $src -t $tgt
T2=$(date +%s)

mamba deactivate

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
