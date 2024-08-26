# CWDE 8/19/24
# translate the en data from Multi30k into another language (es or an indigenous language in our case).
import sys
import argparse
import numpy as np
import transformers

def call_nllb(src_iso = "en", tgt_iso = "es", src_file = "../multi30k-dataset-en/test_2018_flickr.en"):
	# Load model directly, using the HF transformers pipeline
	# Use a pipeline as a high-level helper
	MODEL = "facebook/nllb-200-3.3B"
	DEVICE = "cuda"

	nllb_isos = {"en":"eng_Latn","de":"deu_Latn","fr":"fra_Latn","cs":"ces_Latn","es":"spa_Latn","ay":"ayr_Latn","gn":"grn_Latn","qu":"quy_Latn","zh_hans":"zho_Hans", "zh_hant":"zho_Hant", "ar_arab":"arb_Arab", "ar_latn":"arb_Latn", "uk":"ukr_Cyrl"}

	nllb_src_iso = nllb_isos[src_iso]
	nllb_tgt_iso = nllb_isos[tgt_iso]

	src_txts = np.loadtxt(src_file, dtype="U512", delimiter="|", comments = None) # this delimiter is arbitrary, but without one defined, the delimiter is taken to be any whitespace.

	pipe = transformers.pipeline(f"translation", src_lang = nllb_src_iso, tgt_lang = nllb_tgt_iso, model = MODEL, device=DEVICE, max_length = 512)

	translations = np.array(pipe(src_txts.tolist()))

	result = []
	for translation in translations:
		result.append(translation['translation_text'])

	return result

def savetxt(filename, translations):
	np.savetxt(filename, translations, fmt="%s", delimiter="|")

if __name__ == "__main__":
	# Usage translate_multi30k.py --src en --target es --dataset train
	parser = argparse.ArgumentParser(
		prog = "MULTI30K TRANSLATOR",
		description = "This script translates image descriptions from the original Multi30k datasets into Spanish and three Indigenous languages supported by the NLLB200 project."
	)
	parser.add_argument('-s','--src', choices=['en','fr','de','cs'], default='en', help="the source language for translation from Multi30k. Options: English (en), French (fr), German (de), Czech (cs)")
	parser.add_argument('-t','--target', choices=['es','ay','gn','qu','zh_hans','zh_hant','ar_arab','ar_latn','uk'], default='es', help="the target language for translation from Multi30k. Options: Spanish (es), Aymara (ay), Guarani (gn), and Quechua (qn)")
	parser.add_argument('-d','--dataset', default="test_2018_flickr", help="the multi30k dataset to translate")
	args = parser.parse_args()

	SRC_FILE = f"../multi30k-dataset-{args.src}/{args.dataset}.{args.src}"
	FILE = f"multi30k-dataset-{args.src}-{args.target}/{args.dataset}.{args.target}"

	translations = call_nllb(args.src, args.target, SRC_FILE)
	savetxt(FILE, translations)
