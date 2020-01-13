import re
import os
import argparse
import glob
import numpy as np

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

@registry.register_problem
class FinetuneGeneralProblem(text_problems.Text2TextProblem):

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
        parser.add_argument("--token_err_distribution", default="0.7_0.1_0.1_0.1_0", type=str, help="Space-separated error probabilities in format \"replace insert delete swap recase\".")

        parser.add_argument("--char_err_prob", default=0.05, type=float, help="Probability of char error.")
        parser.add_argument("--char_std_dev", default=0.01, type=float, help="Standard deviation of character error.")
        parser.add_argument("--char_err_distribution", default="0.25_0.25_0.25_0.25_0", type=str, help="Space-separated char-level error probabilities in format \"replace insert delete swap change_diacr\".")
        
        parser.add_argument("--data_ratio", default=1, type=int, help="Ratio of original vs artifical data, i.e. value of 50 means that 50 times more artificial data is used.")
        parser.add_argument("--additional_artificial_sentences", default=0, type=int, help="Number of artificially generated sentences.")
        parser.add_argument("--additional_wiki_sentences", default=0, type=int, help="Number of wiki sentences.")
        parser.add_argument("--additional_data_filtered", default="False", type=str, help="Are additional data filtered or not.")
        
        parser.add_argument("--input_sentence_file", type=str)
        parser.add_argument("--target_sentence_file", type=str)
        
        args = parser.parse_args()

        del data_dir
        del tmp_dir

        # glob pattern specifies path to all chunk files containing synthetised samples
        # TODO specify it to match your situation
        artificial_glob_pattern = '/home/naplava/troja/czesl_experiments/artificial_data/data/{}/chunks/{}-{}-{}-{}/*'.format(args.lang, args.token_err_prob, args.token_err_distribution, args.char_err_prob, args.char_err_distribution)
        artificial_chunks = sorted(glob.glob(artificial_glob_pattern))

        print(dataset_split, type(dataset_split))
        
        if dataset_split == 'train':
            np.random.seed(42)

            original_data = []
            with open(args.input_sentence_file) as f1, open(args.target_sentence_file) as f2:
                for l1, l2 in zip(f1, f2):
                    l1, l2 = l1.strip('\n'), l2.strip('\n')
                    if not l1 or not l2:
                        continue

                    original_data.append((l1, l2))

            artificial_lines = []
            for artificial_chunk in artificial_chunks:
                with open(artificial_chunk) as reader:
                    artificial_lines.extend(reader.read().splitlines())

            num_artificial_sentences = args.additional_artificial_sentences
            num_original_data_cycles_to_generate = max(1, int((num_artificial_sentences / len(original_data)) / args.data_ratio))
            print('Generating {} original lines'.format(num_original_data_cycles_to_generate))
            for _ in range(num_original_data_cycles_to_generate):
                for l1, l2 in original_data:
                    yield {"inputs": l1, "targets": l2}

            print('Generating {} artificial lines.'.format(num_artificial_sentences))
            permutation = np.random.permutation(len(artificial_lines))[:num_artificial_sentences]
            selected_artificial_lines = [artificial_lines[i] for i in permutation]

            artificial_input_sentences, artificial_target_sentences = [], []
            for line in selected_artificial_lines:
                chunks = line.split('\t')
                if len(chunks) < 2:
                    print("Line in artificial data does not contain original and corrected version. Skipping it.")
                    print(chunks)
                    print(line)
                    continue
                yield {"inputs": chunks[1], "targets": chunks[0]}


        else:
            # some non-sense
            yield {"inputs": "unk", "targets": "unk"}

