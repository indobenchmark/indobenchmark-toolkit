# Indobenchmark Toolkit
![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/indobenchmark/indonlg/blob/master/LICENSE) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)

<b>Indobenchmark</b> are collections of Natural Language Understanding (IndoNLU) and Natural Language Generation (IndoNLG) resources for Bahasa Indonesia such as Institut Teknologi Bandung, Universitas Multimedia Nusantara, The Hong Kong University of Science and Technology, Universitas Indonesia, DeepMind, Gojek, and Prosa.AI.

## Toolkit Modules
#### IndoNLGTokenizer
<b>IndoNLGTokenizer</b>  is the tokenizer used by both IndoBART and IndoGPT models. 
The example for using the IndoNLGTokenizer is shown as follow:

- IndoNLGTokenizer for IndoGPT
```python
## Encode ##
from indobenchmark import IndoNLGTokenizer
tokenizer = IndoNLGTokenizer.from_pretrained('indobenchmark/indogpt')
inputs = tokenizer.prepare_input_for_generation('hai, bagaimana kabar', model_type='indogpt', return_tensors='pt')
# inputs: {'input_ids': tensor([[    0,  4693, 39956,  1119,  3447]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]])}

## Decode ##
from indobenchmark import IndoNLGTokenizer
tokenizer = IndoNLGTokenizer.from_pretrained('indobenchmark/indogpt')
text = tokenizer.decode([0,  4693, 39956,  1119,  3447])
# text: '<s> hai, bagaimana kabar'
```

- IndoNLGTokenizer for IndoBART
```python
## Encode ##
from indobenchmark import IndoNLGTokenizer
tokenizer = IndoNLGTokenizer.from_pretrained('indobenchmark/indobart')
inputs = tokenizer.prepare_input_for_generation('hai, bagaimana kabar', return_tensors='pt', 
                       lang_token = '[indonesian]', decoder_lang_token='[indonesian]')
# inputs: {'input_ids': tensor([    0,  4693, 39956,  1119,  3447,     2, 40002]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1])}

## Decode ##
from indobenchmark import IndoNLGTokenizer
tokenizer = IndoNLGTokenizer.from_pretrained('indobenchmark/indobart')
text = tokenizer.decode([0,  4693, 39956,  1119,  3447, 2, 40002])
# text: '<s> hai, bagaimana kabar </s> [indonesian]'
```

**note**: IndoNLGTokenizer will automatically lower case the text input since both the IndoNLGTokenizer, the IndoBart, and the IndoGPT models  are only trained on lower-cased texts.

## Research Paper
IndoNLU has been accepted by AACL-IJCNLP 2020 and you can find the details in our paper https://www.aclweb.org/anthology/2020.aacl-main.85.pdf.
If you are using any component on IndoNLU including Indo4B, FastText-Indo4B, or IndoBERT in your work, please cite the following paper:
```
@inproceedings{wilie2020indonlu,
  title={IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding},
  author={Bryan Wilie and Karissa Vincentio and Genta Indra Winata and Samuel Cahyawijaya and X. Li and Zhi Yuan Lim and S. Soleman and R. Mahendra and Pascale Fung and Syafri Bahar and A. Purwarianti},
  booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
  year={2020}
}
```

IndoNLG has been accepted by EMNLP 2021 and you can find the details in our paper https://arxiv.org/abs/2104.08200.
If you are using any component on IndoNLG including Indo4B-Plus, IndoBART, or IndoGPT in your work, please cite the following paper:
```
@misc{cahyawijaya2021indonlg,
      title={IndoNLG: Benchmark and Resources for Evaluating Indonesian Natural Language Generation}, 
      author={Samuel Cahyawijaya and Genta Indra Winata and Bryan Wilie and Karissa Vincentio and Xiaohong Li and Adhiguna Kuncoro and Sebastian Ruder and Zhi Yuan Lim and Syafri Bahar and Masayu Leylia Khodra and Ayu Purwarianti and Pascale Fung},
      year={2021},
      eprint={2104.08200},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## IndoNLU and IndoNLG Models
### IndoBERT and IndoBERT-lite Models
We provide 4 IndoBERT and 4 IndoBERT-lite Pretrained Language Model [[Link]](https://huggingface.co/indobenchmark)
- IndoBERT-base
  - Phase 1  [[Link]](https://huggingface.co/indobenchmark/indobert-base-p1)
  - Phase 2  [[Link]](https://huggingface.co/indobenchmark/indobert-base-p2)
- IndoBERT-large
  - Phase 1  [[Link]](https://huggingface.co/indobenchmark/indobert-large-p1)
  - Phase 2  [[Link]](https://huggingface.co/indobenchmark/indobert-large-p2)
- IndoBERT-lite-base
  - Phase 1  [[Link]](https://huggingface.co/indobenchmark/indobert-lite-base-p1)
  - Phase 2  [[Link]](https://huggingface.co/indobenchmark/indobert-lite-base-p2)
- IndoBERT-lite-large
  - Phase 1  [[Link]](https://huggingface.co/indobenchmark/indobert-lite-large-p1)
  - Phase 2  [[Link]](https://huggingface.co/indobenchmark/indobert-lite-large-p2)

### FastText (Indo4B)
We provide the full uncased FastText model file (11.9 GB) and the corresponding Vector file (3.9 GB)
- FastText model (11.9 GB) [[Link]](https://storage.googleapis.com/babert-pretraining/IndoNLU_finals/models/fasttext/fasttext.4B.id.300.epoch5.uncased.bin) 
- Vector file (3.9 GB) [[Link]](https://storage.googleapis.com/babert-pretraining/IndoNLU_finals/models/fasttext/fasttext.4B.id.300.epoch5.uncased.vec.zip)

We provide smaller FastText models with smaller vocabulary for each of the 12 downstream tasks
- FastText-Indo4B [[Link]](https://storage.googleapis.com/babert-pretraining/IndoNLU_finals/models/fasttext/fasttext-4B-id-uncased.zip)
- FastText-CC-ID [[Link]](https://storage.googleapis.com/babert-pretraining/IndoNLU_finals/models/fasttext/fasttext-cc-id.zip)

### IndoBART and IndoGPT Models
We provide IndoBART and IndoGPT Pretrained Language Model [[Link]](https://huggingface.co/indobenchmark)
- IndoBART [[Link]](https://huggingface.co/indobenchmark/indobart)
- IndoBART-v2 [[Link]](https://huggingface.co/indobenchmark/indobart-v2)
- IndoGPT2 [[Link]](https://huggingface.co/indobenchmark/indogpt)
