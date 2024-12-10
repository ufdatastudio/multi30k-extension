#!/bin/bash
#SBATCH --job-name=multi30k_kl_div_%j
#SBATCH --output=logs/multi30k_kl_div_%j.out
#SBATCH --error=logs/multi30k_kl_div_%j.err
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

mamba activate test-blank-env

echo "KL Divergence Calculation for Multi30k Translation"
echo "Source Lang(s) = $(echo $src)"
echo "Target Lang(s) = $(echo $tgt)"

T1=$(date +%s)
python multi30k-scripts/similarity_analysis_multi30k.py -s $srcs -t $tgts -o 'kl_div'
T2=$(date +%s)

mamba deactivate

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
