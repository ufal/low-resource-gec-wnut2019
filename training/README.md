# Grammatical Error Correction in Low-Resource Scenarios

This folder stores scripts and data to train GEC system. 

## Requirements

Tensor2Tensor library is used for training neural machine translation system. We made several changes to it so we can set input and target word-dropout rate and also edit-weighted MLE. 
You can install it from https://github.com/arahusky/tensor2tensor (commit 075a590).

## Training models

Each experiment has a configuration file which is passed to [run_training.sh](training/run_training.sh) which starts data conversion (to TFRecords) and afterwards training itself. 

T2T problems specifying data to be saved in T2T records are saved in [problems](training/problems) directory. You should change some paths (marked as TODO) to have it properly working for yourself.

You should also change path to your root directory in your configuration file. Some configuration files are saved in [configs](training/configs) directory.

For start, you can run 
```
bash run_training.sh configs/cs_artificial_errors_config_base_single_gpu.sh
```

which trains Transformer-BASE model on a single GPU for Czech GEC.

## Early stopping 

We did not observe any overfitting when training models on (large enough) synthetic data. 
When fine-tuning with authentic data, overfitting is unfortunately a hot issue. 
In our work, we modified the [run_training.sh](training/run_training.sh) script to save checkpoints more often (~10 minutes interval) and evaluated each checkpoint on development sets to estimate the stopping point.
