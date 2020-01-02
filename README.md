# Grammatical Error Correction in Low-Resource Scenarios

This repository contains information on how to replicate our experiments on training state-of-the-art models for grammatical error correction in low-resource languages. 


## Citation

```
@article{naplava2019grammatical,
  title={Grammatical Error Correction in Low-Resource Scenarios},
  author={N{\'a}plava, Jakub and Straka, Milan},
  booktitle = "Proceedings of the 2019 {EMNLP} Workshop W-{NUT}: The 5th Workshop on Noisy User-generated Text",
  year = "2019",
  address = "Hong Kong",
  publisher = "Association for Computational Linguistics",
}
```
## Requirements

Tensor2Tensor library is used for training neural machine translation system. We updated it so we can set input and target word-dropout rate and also edit-weighted MLE. 
This can be cloned from https://github.com/arahusky/tensor2tensor (commit 075a590).

Additionaly, to generate synthetic data, you need ASpell with dictionary for each language. These can be (on Ubuntu) downloaded as 
```apt-get install aspell-en
apt-get install aspell-cs
apt-get install aspell-ru
apt-get install aspell-en
apt-get install aspell-de
```

## Synthetic data generation 

TODO

## Training models

## Using trained models for correcting text

TODO
