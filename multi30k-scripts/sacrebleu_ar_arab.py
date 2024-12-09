# CWDE615 9-24-24
# Calculate BLEU and ChrF++ scores for the splits of the Arabic dataset
# Must be run as a module with python -m ...
from datasets import load_dataset
import numpy as np
import sacrebleu

bleu = sacrebleu.metrics.BLEU()
chrf = sacrebleu.metrics.CHRF()

dataset_dict = {'train':'train', 'val':'val'}

bleu_scores = []
chrf_scores = []

dir = "multi30k-dataset-en-ar_arab"

for dataset, ar_split in dataset_dict.items():
	ref_ar = np.loadtxt(f'../multi30k-dataset-ar/{ar_split}/Arabic.txt', delimiter = "|", dtype = "U512", comments = None).tolist()
	translation_ar = np.loadtxt(f"multi30k-translations-en/{dir}/{dataset}.ar_arab", delimiter = "|", dtype = "U512", comments = None).tolist()

	bleu_scores.append(f'{dataset} {bleu.corpus_score(translation_ar, ref_ar)}')
	chrf_scores.append(f'{dataset} {chrf.corpus_score(translation_ar, ref_ar)}')

bleu_scores.append(bleu.get_signature())
chrf_scores.append(chrf.get_signature())

np.savetxt(f'{dir}/bleu_analysis_ar_enar.txt', bleu_scores, fmt = '%s')
np.savetxt(f'{dir}/chrf_analysis_ar_enar.txt', chrf_scores, fmt = '%s')

