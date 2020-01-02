token_err_prob=${1:-"0.15"} # probability of introducing error in token
token_err_distribution=${2:-"0.7_0.1_0.1_0.1_0"} # (replace insert delete swap recase)

char_err_prob=${3:-"0.02"} # probability of introducing character error
char_err_distribution=${4:-"0.25_0.25_0.25_0.25_0"} # replace insert delete swap change_diacritics
 
max_chunks=3 # how many parallel jobs to run

monolingual_data=sample_monolingual_data/monolingual_cs.txt
vocabulary=vocabularies/vocabulary_cs.tsv
lang=cs

for i in $(seq 1 $max_chunks); do
	# qsub -q cpu-troja.q -N chunks_cs_$i -j y bash generate_data.sh $i $max_chunks $token_err_prob \"$token_err_distribution\" $char_err_prob \"$char_err_distribution\" $lang $monolingual_data $vocabulary
	bash generate_data.sh $i $max_chunks $token_err_prob $token_err_distribution $char_err_prob $char_err_distribution $lang $monolingual_data $vocabulary &
done

