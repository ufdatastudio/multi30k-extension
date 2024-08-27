# CWDE615 8-26-24
# Retrieves the translations of one of the Multi30k datasets and the dataset itself then performs similarity analysis
# on the data.
import argparse
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

def load_embedding_model():
	# load DistiluseBERT from sentence-transformers
	# see https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
	return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')


def get_embeddings(embedder, sentence_list):
	# retrieve embeddings from DistiluseBERT
	# see https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
	return embedder.encode(sentence_list)


# TODO: Change to a call to similarity method of SentenceTransformer
def get_cosine_similarity(src_embeddings, tgt_embeddings):
	src_tensor = torch.from_numpy(src_embeddings)
	tgt_tensor = torch.from_numpy(tgt_embeddings)

	return F.cosine_similarity(src_tensor, tgt_tensor).numpy()


def write_cosine_similarity(filename, src_embeddings, tgt_embeddings):
	dataset_similarity = get_cosine_similarity(src_embeddings, tgt_embeddings)

	np.savetxt(filename, dataset_similarity, fmt='%s')


def loadtxt(filename):
	return np.loadtxt(filename, dtype = 'U512', delimiter = '|', comments = None)


if __name__ == "__main__":
	# Usage cosine_similarity_multi30k.py --src en --target es
	parser = argparse.ArgumentParser(
	        prog = "MULTI30K TRANSLATION COSINE SIMILARITY CALCULATOR",
	        description = "This script calculates the cosine similarities of image descriptions between emeddings the original Multi30k datasets and its translations into various languages"
	)
	parser.add_argument('-s','--src', choices=['en','fr','de','cs'], default='en', help="the source language for cosine similarity calculation")
	parser.add_argument('-t','--target', choices=['es','ay','gn','qu','zh_hans','zh_hant','ar_arab','ar_latn','uk'], default='es', help="the source language for cosine similarity calculation")

	args = parser.parse_args()

	datasets = np.array(['test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco', 'test_2018_flickr', 'train', 'val'])

	embedder = load_embedding_model()

	src_embeddings_dict = dict()
	uk_embeddings = dict()
	ar_embeddings = dict()

	for dataset in datasets:
		SRC_FILE = f"../multi30k-dataset-{args.src}/{dataset}.{args.src}"
		src_txt = loadtxt(SRC_FILE)
		src_embeddings = get_embeddings(embedder, src_txt)

		TGT_FILE = f"multi30k-dataset-{args.src}-{args.target}/{dataset}.{args.target}"
		tgt_txt = loadtxt(TGT_FILE)
		tgt_embeddings = get_embeddings(embedder, tgt_txt)

		if args.target == 'uk':
			src_embeddings_dict[dataset] = src_embeddings
			uk_embeddings[dataset] = tgt_embeddings

		elif args.target == 'ar_arab':
			src_embeddings_dict[dataset] = src_embeddings
			ar_embeddings[dataset] = tgt_embeddings

		FILE = f"multi30k-dataset-{args.src}-{args.target}/similarity_{dataset}_{args.src}_{args.target}.txt"
		write_cosine_similarity(FILE, src_embeddings, tgt_embeddings)

	if args.target == 'uk':
		# for Ukrainian data we compare our results to Saichyshyna et al.
		# see https://huggingface.co/datasets/turuta/Multi30k-uk
		for dataset in datasets:
			# the val split is not included in the data from Saichyshyna et al.
			if dataset == 'val':
				continue

			# a dict matching the split names from multi30k files to splits defined in Saichyshyna et al.
			# The latter does not use val, so it doesn't appear here
			corr_dict = {'test_2016_flickr':'flickr_2016', 'test_2017_flickr':'flickr_2017', 'test_2018_flickr':'flickr_2018', 'test_2017_mscoco':'mscoco_2017', 'train':'multi30k'}

			# get array from ds. Use the corr_dict to match the corect split name with the dataset at hand.
			turuta_list = load_dataset("turuta/Multi30k-uk", corr_dict[dataset])['train']['uk']
			turuta_array = np.array(turuta_list)

			turuta_embeddings = get_embeddings(embedder, turuta_array)

			FILE = f"multi30k-dataset-{args.src}-{args.target}/similarity_{dataset}_{args.src}_sai.txt"
			write_cosine_similarity(FILE, src_embeddings_dict[dataset], turuta_embeddings)

			FILE = f"multi30k-dataset-{args.src}-{args.target}/similarity_{dataset}_{args.target}_sai.txt"
			write_cosine_similarity(FILE, uk_embeddings[dataset], turuta_embeddings)

	elif args.target == 'ar_arab':
		# for Arabic data we compare our results to the ArEnMulti30k datasets for train and val data.
		# see https://zenodo.org/records/4394718 and https://sites.google.com/view/arenmulti30k
		ar_subset = np.array(['train', 'val'])

		for dataset in ar_subset:
			AR_FILE = f"../multi30k-dataset-ar/{dataset}/Arabic.txt"
			aren_array = loadtxt(AR_FILE)

			aren_embeddings = get_embeddings(embedder, aren_array)

			FILE = f"multi30k-dataset-{args.src}-{args.target}/similarity_{dataset}_{args.src}_enar.txt"
			write_cosine_similarity(FILE, src_embeddings_dict[dataset], aren_embeddings)

			FILE = f"multi30k-dataset-{args.src}-{args.target}/similarity_{dataset}_{args.target}_enar.txt"
			write_cosine_similarity(FILE, ar_embeddings[dataset], aren_embeddings)
