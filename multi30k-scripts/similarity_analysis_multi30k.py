# CWDE615 8-25-24
# Read cosine similarity from file for a language pair. Similarities must be calculated previous to the analysis.
import argparse
import copy
# import matplotlib.pyplot as plt # TODO: Uncomment this and set up matplotlib to make tables and histograms automatically.
import numpy as np

def loadtxt(filename):
	return np.loadtxt(filename, dtype = 'd', delimiter = "|", comments = None)

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


if __name__ == "__main__":
	# Usage similarity_analysis_multi30k.py --srcs "en,fr" --tgts "es,ay,gn,qu"
	parser = argparse.ArgumentParser(
		prog = "MULTI30K COSINE SIMILARITY ANALYZER",
		description = "This script analyzes the cosine similarities written to various files by cosine_similarity_multi30k.py. If the similarity or similarities to be analyzed does/do not exist, it will throw an error."
	)
	parser.add_argument('-s','--srcs', default='cs,de,en,fr', help="comma separated list of ISO 639-1 codes of source languages for cosine similarity analysis. These will be analyzed pairwise with the target languages (tgts).")
	parser.add_argument('-t','--tgts', default='es,ay,gn,qu,zh_hans,zh_hant,ar_arab,uk', help = "comma separated list of ISO 639-1 codes of target languages for cosine similarity analysis. These will be analyzed pairwise with the source languages (srcs).")
	parser.add_argument('-d','--datasets',default='all', help = 'comma separated list of dataset(s) to analyze. Type all as a shortcut to analyze all datasets.')
	parser.add_argument('-a','--analysis',default='all', choices=['mean','hist','all'], help='conduct the mean cosine similarity analysis (mean), a histogram analysis (hist), or all analyzes (all).')

	args = parser.parse_args()

	if args.datasets == 'all':
		datasets = ['test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco', 'test_2018_flickr', 'train', 'val']
	else:
		datasets = args.datasets.strip().split(',')

	src_langs = args.srcs.strip().split(',')
	tgt_langs = args.tgts.strip().split(',')

	if args.analysis == 'mean':
		cosine_similarity_table(src_langs, tgt_langs, datasets)
	elif args.analysis == 'hist':
		cosine_similarity_histograms(src_langs, tgt_langs, datasets)
	elif args.analysis == 'all':
		cosine_similarity_table(src_langs, tgt_langs, datasets)
		cosine_similarity_histograms(src_langs, tgt_langs, datasets)

	# no else block. All options are included in the if and elif blocks.
