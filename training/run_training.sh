set -x

source ~virtualenvs/t2t/bin/activate

configuration_file=$1

source ${configuration_file}
echo "Configuration file loaded"

echo "Creating data directory ${NO_EDIT_DATA_DIR}"
mkdir -p ${NO_EDIT_DATA_DIR}

# echo "Copying vocab"
# if you have custom vocabulary, you can pass it here, so that it is not newly generated (quite slow) 
# cp custom_vocab_file ${NO_EDIT_DATA_DIR}/vocab.$PROBLEM.32768.subwords

echo "Generating data into ${NO_EDIT_DATA_DIR} for problem: ${PROBLEM}"
# generate data
t2t-datagen \
   --t2t_usr_dir="${PROBLEM_DIR}" \
   --data_dir="${NO_EDIT_DATA_DIR}" \
   --tmp_dir=/tmp/${PROBLEM} \
   --problem=$PROBLEM \
   --token_err_prob="$TOKEN_ERR_PROB" \
   --token_err_distribution="$TOKEN_ERR_DISTRIBUTION" \
   --char_err_prob="$CHAR_ERROR_PROB" \
   --char_err_distribution="$CHAR_ERR_DISTRIBUTION" \
   --lang=$LANG

# generate edit weights
VOCAB_FILE=${NO_EDIT_DATA_DIR}/*vocab*
echo "Generating train files with edit-weights into ${DATA_DIR}"
if [ ! -d ${DATA_DIR} ]; then
    mkdir ${DATA_DIR}
    for tf_record in ${NO_EDIT_DATA_DIR}/*-train-*; do
        ~/virtualenvs/t2t/bin/python3 add_weights_to_tfrecord.py ${tf_record} ${DATA_DIR} ${VOCAB_FILE} ${EDIT_WEIGHT}
    done
    
    cp ${VOCAB_FILE} ${DATA_DIR}/$(basename ${VOCAB_FILE})
fi

# train
echo "Training"
t2t-trainer \
  --data_dir=${DATA_DIR} \
  --problem=${PROBLEM} \
  --model=${MODEL} \
  --hparams_set=${MODEL_TYPE} \
  --hparams="input_word_dropout=${INPUT_WORD_DROPOUT_RATE},target_word_dropout=${TARGET_WORD_DROPOUT_RATE},batch_size=${BATCH_SIZE},max_length=${MAX_LEN},learning_rate_warmup_steps=${WARMUP_STEPS},learning_rate_constant=${LEARNING_RATE_CONSTANT},learning_rate_schedule=constant*rsqrt_decay,optimizer=Adafactor" \
  --output_dir=${TRAIN_DIR} \
  --t2t_usr_dir=${PROBLEM_DIR} \
  --worker_gpu=${NUM_GPUS} \
  --train_steps=6000000 \
  --keep_checkpoint_every_n_hours=4 \
  --keep_checkpoint_max=100 \
  --schedule=train \
  --save_checkpoints_secs=3600 
