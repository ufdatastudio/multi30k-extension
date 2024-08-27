# !/bin/bash
# Calculate cosine similarity between the source language in $1 and all relevant target languages.
# Usage: multi30k-scripts/cosine_similarity_all_targets.sh src
for tgt in ar_arab es uk zh_hans zh_hant;
do
        sbatch --export=src=$1,tgt=$tgt multi30k-scripts/multi30k_similarity.sbatch
done
