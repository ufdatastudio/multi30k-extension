# CWDE615
# Script to calculate the COMETkiwi metric for a src-tgt language pair.

from comet import download_model, load_from_checkpoint


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

def run_comet_kiwi(src_list, tgt_list):

	# Get model from HF
	MODEL = "Unbabel/wmt22-cometkiwi-da"
	model_path = download_model(MODEL)
	model = load_from_checkpoint(model_path)

	# Build data list from the translations
	data = build_data_list(src_list, tgt_list)

	# Run model and return output
	model_output = model.predict(data, batch_size=8, gpus=1)
	return model_output

if __name__ == "__main__":
	pass

