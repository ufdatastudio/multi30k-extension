# !/bin/bash
# creates six sbatch translation jobs from the src language to the tgt language
# using the multi30k_translation sbatch script.
# Usage: multi30k-scripts/translate_all_datasets.sh src tgt

for dataset in train test_2016_flickr test_2017_flickr test_2017_mscoco test_2018_flickr val;
do
	sbatch --export=src=$1,tgt=$2,dataset=$dataset multi30k-scripts/multi30k_translation.sbatch
done
