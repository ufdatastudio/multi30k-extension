# !/bin/bash
# Calculate the mean symmetric KL Divergence for each language pair for every dataset..
# Usage: multi30k-scripts/all_kl_div.sh
sbatch --export=srcs="cs,de,en,fr",tgts="ar_arab,es,uk,zh_hans,zh_hant" multi30k-scripts/multi30k_kl_div.sbatch
