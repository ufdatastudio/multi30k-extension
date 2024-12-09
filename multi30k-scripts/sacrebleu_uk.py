# CWDE615 8-29-24
# Calculate BLEU and ChrF++ scores for the splits of the Ukrainian dataset
# Must be run as a module with python -m ...
from datasets import load_dataset
import numpy as np
import sacrebleu

bleu = sacrebleu.metrics.BLEU()
chrf = sacrebleu.metrics.CHRF()

dataset_dict = {'test_2016_flickr':'flickr_2016', 'test_2017_flickr':'flickr_2017', 'test_2018_flickr':'flickr_2018','test_2017_mscoco':'mscoco_2017', 'train':'multi30k'}

bleu_scores = []
chrf_scores = []

dir = "multi30k-dataset-en-uk"

for dataset, sai_split in dataset_dict.items():
	ref_uk = [load_dataset("turuta/Multi30k-uk", sai_split)['train']['uk']]
	translation_uk = np.loadtxt(f"{dir}/{dataset}.uk", delimiter = "|", dtype = "U512", comments = None).tolist()

	bleu_scores.append(f'{dataset} {bleu.corpus_score(translation_uk, ref_uk)}')
	chrf_scores.append(f'{dataset} {chrf.corpus_score(translation_uk, ref_uk)}')

bleu_scores.append(bleu.get_signature())
chrf_scores.append(chrf.get_signature())

np.savetxt(f'{dir}/bleu_analysis_uk_sai.txt', bleu_scores, fmt = '%s')
np.savetxt(f'{dir}/chrf_analysis_uk_sai.txt', chrf_scores, fmt = '%s')

