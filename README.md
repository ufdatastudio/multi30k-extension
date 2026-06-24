# Multi30k Extension
## Description
This repo contains the code and raw data used to create and evaluate the MultiScript30k dataset we propose in [MultiScript30k: Leveraging Multilingual Embeddings to Extend Cross Script Parallel Data](https://arxiv.org/abs/2512.11074).
The data in this repository is machine translated from the [Multi30k](https://github.com/multi30k/dataset) dataset into the languages indicated by the directory names and file extensions.

This repo exists to facilitate reproduction of our analysis and the creation of new synthetic Multi30k extensions.
To simply access our dataset, please see the [MultiScript30k repo](https://github.com/ufdatastudio/multiscript30k/tree/main).

If you find this data useful in your work, please consider citing our publication and the original Multi30k papers from the WMT shared tasks on multimodal machine translation.
```
% Our paper, MultiScript30k: Leveraging Multilingual Embeddings to Extend Cross Script Parallel Data
@misc{driggersellis2025multiscript30kleveragingmultilingualembeddings,
      title={MultiScript30k: Leveraging Multilingual Embeddings to Extend Cross Script Parallel Data}, 
      author={Christopher Driggers-Ellis and Detravious Brinkley and Ray Chen and Aashish Dhawan and Daisy Zhe Wang and Christan Grant},
      year={2025},
      eprint={2512.11074},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.11074}, 
}

% Original Multi30k-En and De datasets
@InProceedings{W16-3210,
  author = 	"Elliott, Desmond
		and Frank, Stella
		and Sima'an, Khalil
		and Specia, Lucia",
  title = 	"Multi30K: Multilingual English-German Image Descriptions",
  booktitle = 	"Proceedings of the 5th Workshop on Vision and Language",
  year = 	"2016",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"70--74",
  location = 	"Berlin, Germany",
  doi = 	"10.18653/v1/W16-3210",
  url = 	"http://www.aclweb.org/anthology/W16-3210"
}

% Original Multi30k-Fr and 2017 test data
@InProceedings{elliott-EtAl:2017:WMT,
  author    = {Elliott, Desmond  and  Frank, Stella  and  Barrault, Lo\"{i}c  and  Bougares, Fethi  and  Specia, Lucia},
  title     = {Findings of the Second Shared Task on Multimodal Machine Translation and Multilingual Image Description},
  booktitle = {Proceedings of the Second Conference on Machine Translation, Volume 2: Shared Task Papers},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics},
  pages     = {215--233},
  url       = {http://www.aclweb.org/anthology/W17-4718}
}

% Original Multi30k-Cs and 2018 test data
@inproceedings{barrault2018findings,
  title={Findings of the Third Shared Task on Multimodal Machine Translation},
  author={Barrault, Lo{\"\i}c and Bougares, Fethi and Specia, Lucia and Lala, Chiraag and Elliott, Desmond and Frank, Stella},
  booktitle={Proceedings of the Third Conference on Machine Translation: Shared Task Papers},
  pages={304--323},
  year={2018}
}
```
