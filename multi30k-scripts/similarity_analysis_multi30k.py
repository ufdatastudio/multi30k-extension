# CWDE615 8-25-24
# Read cosine similarity from file for a language pair. Similarities must be calculated previous to the analysis.
import argparse
import copy
# import matplotlib.pyplot as plt # TODO: Uncomment this and set up matplotlib to make tables and histograms automatically.
from cosine_similarity_multi30k import load_embedding_model, get_embeddings
import numpy as np
from torch import from_numpy
import torch.nn.functional as F

def loadtxt(filename):
	return np.loadtxt(filename, dtype = 'd', delimiter = "|", comments = None)

def loadtensor(filename):
	np_array = loadtxt(filename)
	return from_numpy(np_array)

def get_lang_dirs(src_lang, tgt_lang, src_sub = 'en'):
	src_lang_dir = src_lang
	tgt_lang_dir = tgt_lang

	if src_lang not in ['en','fr','de','cs']:
		src_lang_dir = src_sub

	if tgt_lang == 'enar':
		tgt_lang_dir = 'ar_arab'
	elif tgt_lang == 'sai':
		tgt_lang_dir = 'uk'

	return src_lang_dir, tgt_lang_dir


def get_mean_cosine_similarity(src_lang, tgt_lang, datasets):
	sums = []
	means = []
	num = 0

	src_lang_dir, tgt_lang_dir = get_lang_dirs(src_lang, tgt_lang)

	for dataset in datasets:
		if src_lang == 'cs' and dataset in ['test_2017_flickr', 'test_2017_mscoco']:
			continue

		IN_FILE = f"multi30k-dataset-{src_lang_dir}-{tgt_lang_dir}/similarity_{dataset}_{src_lang}_{tgt_lang}.txt"
		sims = loadtxt(IN_FILE)
		sums.append(sims.sum())
		means.append(sims.mean())
		num += sims.shape[0]

	OUT_FILE = f"tables/cosine_similarity_table_{src_lang}_{tgt_lang}.txt"
	np.savetxt(OUT_FILE, list(zip(datasets, means)), fmt='%s')

	return np.array(sums).sum() / num

def cosine_similarity_table(src_langs, tgt_langs, datasets):
	pair_means = dict()

	for src_lang in src_langs:
		for tgt_lang in tgt_langs:
			pair_means[f'{src_lang}-{tgt_lang}'] = get_mean_cosine_similarity(src_lang, tgt_lang, datasets)

	OUT_FILE = f'tables/cosine_similarity_table_overall.txt'
	results = np.empty(shape = (len(src_langs) * len(tgt_langs), 2), dtype = 'U20')
	results[:, 0] = list(pair_means.keys())
	results[:, 1] = list(pair_means.values())
	np.savetxt(OUT_FILE, results, fmt = '%s')


def get_cosine_similarity_histogram(src_lang, tgt_lang, datasets, bins = 10):
	histogram = np.zeros(shape = (bins,), dtype = 'int')

	src_lang_dir, tgt_lang_dir = get_lang_dirs(src_lang, tgt_lang)

	for dataset in datasets:
		if src_lang == 'cs' and dataset in ['test_2017_flickr', 'test_2017_mscoco']:
			continue

		IN_FILE = f"multi30k-dataset-{src_lang_dir}-{tgt_lang_dir}/similarity_{dataset}_{src_lang}_{tgt_lang}.txt"
		sims = loadtxt(IN_FILE)
		histogram += np.histogram(sims, range=(0,1), bins = bins)[0]

		OUT_FILE = f"multi30k-dataset-{src_lang_dir}-{tgt_lang_dir}/histogram_{dataset}_{src_lang}_{tgt_lang}.txt"
		np.savetxt(OUT_FILE, histogram, fmt='%s')

	OUT_FILE = f"multi30k-dataset-{src_lang_dir}-{tgt_lang_dir}/histogram_{src_lang}_{tgt_lang}.txt"
	np.savetxt(OUT_FILE, histogram, fmt='%s')


def cosine_similarity_histograms(src_langs, tgt_langs, datasets, bins = 10):
	for src_lang in src_langs:
		for tgt_lang in tgt_langs:
			get_cosine_similarity_histogram(src_lang, tgt_lang, datasets, bins)

def sym_kl_div(src, tgt):
	src_tensor = from_numpy(src)
	tgt_tensor = from_numpy(tgt)

	left = F.kl_div(src_tensor, tgt_tensor, reduction = 'batchmean')
	right = F.kl_div(tgt_tensor, src_tensor, reduction = 'batchmean')

	return left + right

def get_mean_kl_div(embedder, src_lang, tgt_lang, datasets):

	divs = []

	for dataset in datasets:
		if src_lang == 'cs' and dataset in ['test_2017_flickr', 'test_2017_mscoco']:
			continue

		SRC_FILE = f"../multi30k-dataset-{args.src}/{dataset}.{args.src}"
		src_txt = loadtxt(SRC_FILE)
		src_embeddings = get_embeddings(embedder, src_txt)

		TGT_FILE = f"multi30k-dataset-{args.src}-{args.target}/{dataset}.{args.target}"
		tgt_txt = loadtxt(TGT_FILE)
		tgt_embeddings = get_embeddings(embedder, tgt_txt)

		kl_div = sym_kl_div(src_embeddings, tgt_embeddings)
		divs.append(kl_div)

	OUT_FILE = f"tables/kl_div_{src_lang}_{tgt_lang}.txt"
	np.savetxt(OUT_FILE, list(zip(datasets, divs)), fmt='%s')

	return np.array(divs).mean()

def kl_div_table(src_langs, tgt_langs, datasets):
	pair_means = dict()
	model = load_embedding_model()

	for src_lang in src_langs:
		for tgt_lang in tgt_langs:
			pair_means[f'{src_lang}-{tgt_lang}'] = get_mean_kl_div(model, src_lang, tgt_lang, datasets)

	OUT_FILE = f'tables/kl_div_table_overall.txt'
	results = np.empty(shape = (len(src_langs) * len(tgt_langs), 2), dtype = 'U20')
	results[:, 0] = list(pair_means.keys())
	results[:, 1] = list(pair_means.values())
	np.savetxt(OUT_FILE, results, fmt = '%s')


if __name__ == "__main__":
	# Usage similarity_analysis_multi30k.py --srcs "en,fr" --tgts "es,ay,gn,qu" --operation "kl_div"
	parser = argparse.ArgumentParser(
		prog = "MULTI30K SIMILARITY ANALYZER",
		description = "This script analyzes the KL divergence of encodings and cosine similarities written to various files by kl_div_multi30k.py and cosine_similarity_multi30k.py, respectively. If the similarity or similarities to be analyzed does/do not exist, it will throw an error."
	)
	parser.add_argument('-s','--srcs', default='cs,de,en,fr', help="comma separated list of ISO 639-1 codes of source languages for cosine similarity analysis. These will be analyzed pairwise with the target languages (tgts).")
	parser.add_argument('-t','--tgts', default='es,ay,gn,qu,zh_hans,zh_hant,ar_arab,uk', help = "comma separated list of ISO 639-1 codes of target languages for cosine similarity analysis. These will be analyzed pairwise with the source languages (srcs).")
	parser.add_argument('-d','--datasets',default='all', help = 'comma separated list of dataset(s) to analyze. Type all as a shortcut to analyze all datasets.')
	parser.add_argument('-a','--analysis',default='all', choices=['mean','hist','all'], help='conduct the mean cosine similarity analysis (mean), a histogram analysis (hist), or all analyzes (all). Ignored when operation argument is set to "kl_div," which only performs tabular analysis.')
	parser.add_argument('-o','--operation', default='cosine', choices=['cosine','kl_div'], help='which similarity metric to calculate.')

	args = parser.parse_args()

	if args.datasets == 'all':
		datasets = ['test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco', 'test_2018_flickr', 'train', 'val']
	else:
		datasets = args.datasets.strip().split(',')

	src_langs = args.srcs.strip().split(',')
	tgt_langs = args.tgts.strip().split(',')

	if args.operation == 'kl_div':
		kl_div_table(src_langs, tgt_langs, datasets)
	elif args.analysis == 'mean': # no need to check beyond this point for args.operation. It is guaranteed to be 'cosine.'
		cosine_similarity_table(src_langs, tgt_langs, datasets)
	elif args.analysis == 'hist':
		cosine_similarity_histograms(src_langs, tgt_langs, datasets)
	elif args.analysis == 'all':
		cosine_similarity_table(src_langs, tgt_langs, datasets)
		cosine_similarity_histograms(src_langs, tgt_langs, datasets)

	# no else block. All options are included in the if and elif blocks.
