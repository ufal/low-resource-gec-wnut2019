import re
import os
import argparse
import glob

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

@registry.register_problem
class ArtificialErrors(text_problems.Text2TextProblem):

    @property
    def approx_vocab_size(self):
        return 2 ** 15 # ~32k

    @property
    def is_generate_per_split(self):
        # custom train/test split
        return True

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--t2t_usr_dir", type=str)
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--tmp_dir", type=str)
        parser.add_argument("--problem", type=str)
        parser.add_argument("--lang", type=str)
        parser.add_argument("--token_err_prob", default=0.15, type=float, help="Probability of token error.")
        parser.add_argument("--token_std_dev", default=0.2, type=float, help="Standard deviation of token error.")
        parser.add_argument("--token_err_distribution", default="0.7_0.1_0.1_0.1", type=str, help="Space-separated error probabilities in format \"replace insert delete swap\".")

        parser.add_argument("--char_err_prob", default=0.05, type=float, help="Probability of char error.")
        parser.add_argument("--char_std_dev", default=0.01, type=float, help="Standard deviation of character error.")
        parser.add_argument("--char_err_distribution", default="0.25_0.25_0.25_0.25_0", type=str, help="Space-separated char-level error probabilities in format \"replace insert delete swap change_diacr\".")
        args = parser.parse_args()
        del data_dir
        del tmp_dir

        # glob pattern specifies path to all chunk files containing synthetised samples
        # TODO specify it to match your situation
        glob_pattern = '/home/naplava/troja/czesl_experiments/artificial_data/data/{}/chunks/{}-{}-{}-{}/*'.format(args.lang, args.token_err_prob, args.token_err_distribution, args.char_err_prob, args.char_err_distribution)
        print('glob_pattern:' + glob_pattern)
        train_files = glob.glob(glob_pattern)
        if dataset_split == 'train':
            for train_file in train_files:
                print(train_file)
                with open(train_file) as f:
                    for line in f:
                        line = line.strip('\n')
                        if not line:
                            continue
                        chunks = line.split('\t')
                        if len(chunks) < 2:
                        yield {"inputs": chunks[1], "targets": chunks[0]}
        
        else:
            yield {"inputs": "unk", "targets": "unk"}


