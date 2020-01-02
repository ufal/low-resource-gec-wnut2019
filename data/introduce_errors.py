import argparse
import fileinput
import string

import aspell
import numpy as np


allowed_source_delete_tokens = [',', '.', '!', '?']

czech_diacritics_tuples = [('a', 'á'), ('c', 'č'), ('d', 'ď'), ('e', 'é', 'ě'), ('i', 'í'), ('n', 'ň'), ('o', 'ó'), ('r', 'ř'), ('s', 'š'),
                           ('t', 'ť'), ('u', 'ů', 'ú'), ('y', 'ý'), ('z', 'ž')]
czech_diacritizables_chars = [char for sublist in czech_diacritics_tuples for char in sublist] + [char.upper() for sublist in
                                                                                                  czech_diacritics_tuples for char in
                                                                                                  sublist]


def get_char_vocabulary(lang):
    if lang == 'cs':
        czech_chars_with_diacritics = 'áčďěéíňóšřťůúýž'
        czech_chars_with_diacritics_upper = czech_chars_with_diacritics.upper()
        allowed_chars = ', .'
        allowed_chars += string.ascii_lowercase + string.ascii_uppercase + czech_chars_with_diacritics + czech_chars_with_diacritics_upper
        return list(allowed_chars)
    elif lang == 'en':
        allowed_chars = ', .'
        allowed_chars += string.ascii_lowercase + string.ascii_uppercase
        return list(allowed_chars)
    elif lang == 'de':
        german_special = 'ÄäÖöÜüẞß'
        allowed_chars = ', .'
        allowed_chars += string.ascii_lowercase + string.ascii_uppercase + german_special
        return list(allowed_chars)
    elif lang == 'ru':
        russian_special = 'бвгджзклмнпрстфхцчшщаэыуояеёюий'
        russian_special += russian_special.upper()
        russian_special += 'ЬьЪъ'
        allowed_chars = ', .'
        allowed_chars += russian_special
        return list(allowed_chars)


def get_token_vocabulary(tsv_token_file):
    tokens = []
    with open(tsv_token_file) as reader:
        for line in reader:
            line = line.strip('\n')
            token, freq = line.split('\t')

            if token.isalpha():
                tokens.append(token)

    return tokens


def introduce_token_level_errors_on_sentence(tokens, replace_prob, insert_prob, delete_prob, swap_prob, recase_prob, err_prob, std_dev,
                                             word_vocabulary, aspell_speller):
    num_errors = int(np.round(np.random.normal(err_prob, std_dev) * len(tokens)))
    num_errors = min(max(0, num_errors), len(tokens))  # num_errors \in [0; len(tokens)]

    if num_errors == 0:
        return ' '.join(tokens)
    token_ids_to_modify = np.random.choice(len(tokens), num_errors, replace=False)

    new_sentence = ''
    for token_id in range(len(tokens)):
        if token_id not in token_ids_to_modify:
            if new_sentence:
                new_sentence += ' '
            new_sentence += tokens[token_id]
            continue

        current_token = tokens[token_id]
        operation = np.random.choice(['replace', 'insert', 'delete', 'swap', 'recase'], p=[replace_prob, insert_prob, delete_prob,
                                                                                           swap_prob, recase_prob])
        new_token = ''
        if operation == 'replace':
            if not current_token.isalpha():
                new_token = current_token
            else:
                proposals = aspell_speller.suggest(current_token)[:10]
                if len(proposals) > 0:
                    new_token = np.random.choice(proposals)  # [np.random.randint(0, len(proposals))]
                else:
                    new_token = current_token
        elif operation == 'insert':
            new_token = current_token + ' ' + np.random.choice(word_vocabulary)
        elif operation == 'delete':
            if not current_token.isalpha() or current_token in allowed_source_delete_tokens:
                new_token = current_token
            else:
                new_token = ''
        elif operation == 'recase':
            if not current_token.isalpha():
                new_token = current_token
            elif current_token.islower():
                new_token = current_token[0].upper() + current_token[1:]
            else:
                # either whole word is upper-case or mixed-case
                if np.random.random() < 0.5:
                    new_token = current_token.lower()
                else:
                    num_recase = min(len(current_token), max(1, int(np.round(np.random.normal(0.3, 0.4) * len(current_token)))))
                    char_ids_to_recase = np.random.choice(len(current_token), num_recase, replace=False)
                    new_token = ''
                    for char_i, char in enumerate(current_token):
                        if char_i in char_ids_to_recase:
                            if char.isupper():
                                new_token += char.lower()
                            else:
                                new_token += char.upper()
                        else:
                            new_token += char

        elif operation == 'swap':
            if token_id == len(tokens) - 1:
                continue

            new_token = tokens[token_id + 1]
            tokens[token_id + 1] = tokens[token_id]

        if new_sentence and new_token:
            new_sentence += ' '
        new_sentence = new_sentence + new_token

    return new_sentence


def introduce_char_level_errors_on_sentence(sentence, replace_prob, insert_prob, delete_prob, swap_prob, change_diacritics_prob, err_prob,
                                            std_dev, char_vocabulary):
    sentence = list(sentence)
    num_errors = int(np.round(np.random.normal(err_prob, std_dev) * len(sentence)))
    num_errors = min(max(0, num_errors), len(sentence))  # num_errors \in [0; len(sentence)]

    if num_errors == 0:
        return ''.join(sentence)

    char_ids_to_modify = np.random.choice(len(sentence), num_errors, replace=False)

    new_sentence = ''
    for char_id in range(len(sentence)):
        if char_id not in char_ids_to_modify:
            new_sentence += sentence[char_id]
            continue

        operation = np.random.choice(['replace', 'insert', 'delete', 'swap', 'change_diacritics'], 1,
                                     p=[replace_prob, insert_prob, delete_prob, swap_prob, change_diacritics_prob])

        current_char = sentence[char_id]
        new_char = ''
        if operation == 'replace':
            if current_char.isalpha():
                new_char = np.random.choice(char_vocabulary)
            else:
                new_char = current_char
        elif operation == 'insert':
            new_char = current_char + ' ' + np.random.choice(char_vocabulary)
        elif operation == 'delete':
            if current_char.isalpha():
                new_char = ''
            else:
                new_char = current_char
        elif operation == 'swap':
            if char_id == len(sentence) - 1:
                continue

            new_char = sentence[char_id + 1]
            sentence[char_id + 1] = sentence[char_id]
        elif operation == 'change_diacritics':
            if current_char in czech_diacritizables_chars:
                is_lower = current_char.islower()
                current_char = current_char.lower()
                char_diacr_group = [group for group in czech_diacritics_tuples if current_char in group][0]
                new_char = np.random.choice(char_diacr_group)

                if not is_lower:
                    new_char = new_char.upper()

        new_sentence += new_char

    return new_sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("token_file", type=str, help="TSV file with tokens.")

    parser.add_argument("--lang", type=str, default="cs", help="Language identifier for ASpell (e.g. cs, en, de, ru).")

    parser.add_argument("--token_err_prob", default=0.15, type=float, help="Probability of token error.")
    parser.add_argument("--token_std_dev", default=0.2, type=float, help="Standard deviation of token error.")
    parser.add_argument("--token_err_distribution", default="0.7_0.1_0.1_0.1_0", type=str,
                        help="Space-separated error probabilities in format "
                             "\"replace insert delete swap recase\".")

    parser.add_argument("--char_err_prob", default=0.02, type=float, help="Probability of char error.")
    parser.add_argument("--char_std_dev", default=0.01, type=float, help="Standard deviation of character error.")
    parser.add_argument("--char_err_distribution", default="0.2_0.2_0.2_0.2_0.2", type=str,
                        help="Space-separated char-level error probabilities in format \"replace insert delete swap change_diacritics\".")

    args = parser.parse_args()

    token_err_distribution = args.token_err_distribution.split('_')
    if len(token_err_distribution) != 5:
        raise ValueError('You must provide exactly five floats!, provided: {}'.format(token_err_distribution))

    token_replace_prob, token_insert_prob, token_delete_prob, token_swap_prob, recase_prob = map(float, token_err_distribution)
    if not np.isclose(token_replace_prob + token_insert_prob + token_delete_prob + token_swap_prob + recase_prob, 1.):
        raise ValueError('Provided token error probabilites must sum up to 1. They currently sum up to {}'.format(
            str(token_replace_prob + token_insert_prob + token_delete_prob + token_swap_prob + recase_prob)))

    char_err_distribution = args.char_err_distribution.split('_')
    if len(char_err_distribution) != 5:
        raise ValueError('You must provide exactly five floats!, provided {}'.format(char_err_distribution))

    char_replace_prob, char_insert_prob, char_delete_prob, char_swap_prob, change_diacritics_prob = map(float, char_err_distribution)
    if not np.isclose(char_replace_prob + char_insert_prob + char_delete_prob + char_swap_prob + change_diacritics_prob, 1.):
        raise ValueError('Provided character error probabilites must sum up to 1. They currently sum up to {}'.format(
            str(char_replace_prob + char_insert_prob + char_delete_prob + char_swap_prob + change_diacritics_prob)))

    tokens = get_token_vocabulary(args.token_file)
    characters = get_char_vocabulary(args.lang)
    aspell_speller = aspell.Speller('lang', args.lang)

    for line in fileinput.input(('-',)):  # read from std.in (otherwise from files provided as arguments)
        input_line = line = line.strip('\n')
        # introduce word-level errors
        line = introduce_token_level_errors_on_sentence(line.split(' '), token_replace_prob, token_insert_prob, token_delete_prob,
                                                        token_swap_prob, recase_prob, float(args.token_err_prob), float(args.token_std_dev),
                                                        tokens, aspell_speller)

        if '\t' in line or '\n' in line:
            raise ValueError('!!! Error !!! ' + line)

        # introduce spelling errors
        line = introduce_char_level_errors_on_sentence(line, char_replace_prob, char_insert_prob, char_delete_prob, char_swap_prob,
                                                       change_diacritics_prob, float(args.char_err_prob), float(args.char_std_dev),
                                                       characters)

        print('{}\t{}'.format(input_line, line))
