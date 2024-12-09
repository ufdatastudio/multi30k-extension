# CWDE615 12/8/24
# Script to calculate the COMETkiwi metric for a src-tgt language pair.

import argparse
import numpy as np
from comet import download_model, load_from_checkpoint

def loadtxt(filename):
	return np.loadtxt(filename, dtype = 'str', delimiter = "|", comments = None)


def load_src_tgt_lists(src_lang, tgt_lang):
	IN_FILE_SRC = f"../multi30k-dataset-{src_lang}/{dataset}.{src_lang}"
	src_list = loadtxt(IN_FILE_SRC)

	IN_FILE_TGT = f"multi30k-translations-{src_lang}/multi30k-dataset-{src_lang}-{tgt_lang}/{dataset}.{tgt_lang}"
	tgt_list = loadtxt(IN_FILE_TGT)

	return src_list, tgt_list


def build_dict_pair(src_txt, tgt_txt):
	return {
		"src": src_txt,
		"mt": tgt_txt
	}


def build_data_list(src_list, tgt_list):
	data = []
	for src_txt, tgt_txt in zip(src_list, tgt_list):
		data.append(build_dict_pair(src_txt, tgt_txt))

	return data


def load_comet_model():
	# Get model from HF
	MODEL = "Unbabel/wmt23-cometkiwi-da-xl"
	model_path = download_model(MODEL)

	return load_from_checkpoint(model_path)


def run_comet_kiwi(model, src_list, tgt_list):

	# Build data list from the translations
	data = build_data_list(src_list, tgt_list)

	# Run model and return output
	model_output = model.predict(data, batch_size=8, gpus=1)
	return model_output


if __name__ == "__main__":
		# Usage similarity_analysis_multi30k.py --srcs "cs,de,en,fr" --tgts "ar_arab,es,uk,zh_hans,zh_hant"
	parser = argparse.ArgumentParser(
		prog = "MULTI30K COMETKIWI SCORE CALCULATOR",
		description = "This script analyzes translation quality of source texts from Multi30k and corresponding machine translations in the tgt languages."
	)
	parser.add_argument('-s','--srcs', default='cs,de,en,fr', help="comma separated list of ISO 639-1 codes of source languages for cosine similarity analysis. These will be analyzed pairwise with the target languages (tgts).")
	parser.add_argument('-t','--tgts', default='es,zh_hans,zh_hant,ar_arab,uk', help = "comma separated list of ISO 639-1 codes of target languages for cosine similarity analysis. These will be analyzed pairwise with the source languages (srcs).")
	parser.add_argument('-d','--datasets',default='all', help = 'comma separated list of dataset(s) to analyze. Type all as a shortcut to analyze all datasets.')

	args = parser.parse_args()

	if args.datasets == 'all':
		datasets = ['test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco', 'test_2018_flickr', 'train', 'val']
	else:
		datasets = args.datasets.strip().split(',')

	src_langs = args.srcs.strip().split(',')
	tgt_langs = args.tgts.strip().split(',')

	comet_scores = []

	model = load_comet_model()

	for src_lang in src_langs:
		for tgt_lang in tgt_langs:
			for dataset in datasets:
				src_list, tgt_list = load_src_tgt_lists(src_lang, tgt_lang)
				comet_scores.append({"src": src_lang, "tgt": tgt_lang, "dataset": dataset, "score": run_comet_kiwi(model, src_list, tgt_list)['system_score']})

	print(comet_scores)
