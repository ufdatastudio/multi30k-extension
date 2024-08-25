# CWDE615 8-25-24
# Read cosine similarity from file for a language pair. Similarities must be calculated previous to the analysis.
import argparse
# import matplotlib.pyplot as plt # TODO: Uncomment this and set up matplotlib to make tables and histograms automatically.
import numpy

def get_mean_cosine_similarity(src_lang, tgt_lang, datasets):
	accumulator = []

	for dataset in datasets:
                IN_FILE = f"multi30k-dataset-{src_lang}-{tgt_lang}/similarity_{dataset}_{src_lang}_{tgt_lang}.txt"
		sims = np.loadtxt(IN_FILE, dtype = 'd')
		accumulator.append(sims)

	return np.array(accumulator).mean() # make the list a 2D matrix and compute mean of the flattened array (default behavior)


def cosine_similarity_table(src_langs, tgt_langs, datasets):
	pair_means = dict()

	for src_lang in src_langs:
                for tgt_lang in tgt_langs:
			pair_means[f'{src_lang}-{tgt_lang}'] = get_mean_cosine_similarity(src_lang, tgt_lang, datasets)

	OUT_FILE = f'tables/cosine_similarity_table.txt'
	np.savetxt(OUT_FILE, np.array([pair_means.keys(), pair_means.values()]).T, fmt = '%s')


def get_cosine_similarity_histogram(src_lang, tgt_lang, datasets, bins = 10)
	histogram = np.zeros(shape = (bins,), dtype = 'int')
	bins_array = np.arange(0, 1, 1 / bins)

	for dataset in datasets:
		IN_FILE = f"multi30k-dataset-{src_lang}-{tgt_lang}/similarity_{dataset}_{src_lang}_{tgt_lang}.txt"
		sims = np.loadtxt(IN_FILE, dtype='d')
		histograms += np.histograms(sims, bins = bins_array)

	OUT_FILE = f"multi30k-dataset-{args.src}-{args.target}/histogram_{src_lang}_{tgt_lang}.txt"
	np.savetxt(OUT_FILE, histogram, fmt='%s')


def cosine_similarity_histograms(src_langs, tgt_langs, datasets, bins = 10)
	for src_lang in src_langs:
		for tgt_lang in tgt_langs:
			get_cosine_similarity_histogram(src_lang, tgt_lang, datasets, bins)


if __name__ == "__main__":
	# Usage similarity_analysis_multi30k.py --srcs "en,fr" --tgts "es,ay,gn,qu"
	parser = argparse.ArgumentParser(
		prog = "MULTI30K COSINE SIMILARITY ANALYZER"
		descriptions = "This script analyzes the cosine similarities written to various files by cosine_similarity_multi30k.py. If the similarity or similarities to be analyzed does/do not exist, it will throw an error."
	)
	parser.add_argument('-s','--srcs', default='cs,de,en,fr', help="comma separated list of ISO 639-1 codes of source languages for cosine similarity analysis. These will be analyzed pairwise with the target languages (tgts).")
	parser.add_argument('-t','--tgts', default='es,ay,gn,qu,zh_hans,zh_hant,ar_arab,ar_latn,uk', help = "comma separated list of ISO 639-1 codes of target languages for cosine similarity analysis. These will be analyzed pairwise with the source languages (srcs)."
	parser.add_argument('-d','--datasets',default='all', help = 'comma separated list of dataset(s) to analyze. Type all as a shortcut to analyze all datasets.'
	parser.add_argument('-a','--analysis',default='all', choices=['mean','hist','all'], help='conduct the mean cosine similarity analysis (mean), a histogram analysis (hist), or all analyzes (all).')

	args = parser.parse_args()cosine_similarity_histogram(src_langs, tgt_langs)

	if args.dataset == 'all':
		datasets = np.array(['test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco', 'test_2018_flickr', 'train', 'val'])
	else
		datasets = args.dataset.strip().split(',')

	src_langs = args.srcs.strip().split(',')
	tgt_langs = args.tgts.strip().split(',')

	if args.analysis == 'mean':
		cosine_similarity_table(src_langs, tgt_langs, datasets)
	elif args.analysis == 'hist':
		cosine_similarity_histogram(src_langs, tgt_langs, datasets)
	elif args.analysis == 'all':
		cosine_similarity_table(src_langs, tgt_langs, datasets)
		cosine_similarity_histogram(src_langs, tgt_langs, datasets)

	# no else block. All options are included in the if and elif blocks.
