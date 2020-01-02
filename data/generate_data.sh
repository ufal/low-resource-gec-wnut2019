source ~/virtualenvs/aspell/bin/activate # virtualenv with all requirements (mainly aspell-python-py3)

set -ex

chunk_number=$1
max_chunks=$2

token_err_prob=$3
token_err_distribution=$4

char_err_prob=$5
char_err_distribution=$6

lang=$7
monolingual_data=$8
vocabulary=$9

dirname=chunks/$lang/$token_err_prob-$token_err_distribution-$char_err_prob-$char_err_distribution # dirname where to save generated data
mkdir -p "$dirname"

split --number=l/$chunk_number/$max_chunks $monolingual_data | python3 introduce_errors.py $vocabulary --lang=$lang --token_err_prob=$token_err_prob --token_err_distribution=$token_err_distribution --char_err_prob=$char_err_prob --char_err_distribution=$char_err_distribution > "$dirname/chunk_$chunk_number-$max_chunks.txt"

