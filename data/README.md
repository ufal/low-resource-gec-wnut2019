# Grammatical Error Correction in Low-Resource Scenarios

This folder stores scripts and data to generate synthetic data.

If you are interested in authentic parallel data, you can find them on following links:
- Czech [AKCES-GEC](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3057)
- German [FALKO-MERLIN GEC corpus](http://www.sfs.uni-tuebingen.de/~adriane/download/wnut2018/data.tar.gz)
- Russian [RULEC-GEC](https://github.com/arozovskaya/RULEC-GEC) -- upon request
- English - there are several datasets, most of which can be found on https://www.cl.cam.ac.uk/research/nl/bea2019st/#data

## Requirements

- ASpell dictionaries, which can be (on Ubuntu) downloaded as
```
apt-get install aspell-cs
apt-get install aspell-ru
apt-get install aspell-en
apt-get install aspell-de
```

- [generate_data.sh](data/generate_data.sh) script activates environment with Python3 and supposes it to contain all packages from [requirements.txt](data/requirements.txt).
So either modify the script for your needs or run
```
python3 -m venv ~/virtualenvs/aspell
pip install -r requirements.txt
```

## Generating Synthetic Data

Data are generated using [generate_data_wrapper.sh](data/generate_data_wrapper.sh) script. This script stores several variables:
- character/token level corruption probabilities 
- language (for ASpell)
- path to clean monolingual data file (one sentence per line)
- path to vocabulary tsv file (token\toccurence format)
- max_chunks -- how many parallel jobs to run (to speed it up)

It outputs both original and corrupted sentence separated with tabulator. If you use multiple parallel jobs, you should concatenate resulting files into one file.

[Vocabularies](data/vocabularies) directory contains vocabulary files used in our experiments. [Sample_monolingual_data](data/sample_monolingual_data) then contains a file with 10 000 clean Czech sentences. In our experiments, we used WMT News Crawl Data (http://data.statmt.org/news-crawl/) for each language.
