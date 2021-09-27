# !/bin/bash

basepath=$(cd `dirname $0`; pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath

test_anwser_candidates=$PREDICT_FILE_DIR/answer_candidates.json
bert_dir='/read-only/common/pretrain_model/transformers/bert-base-chinese/'
saved_file=${SAVE_MODEL_DIR}
model='bert_encoder_match.baseline_lmcl'

python server.py --do_eval --model ${model} --test_anwser_candidates ${test_anwser_candidates} --bert_dir ${bert_dir} --dialog_encoder_file ${bert_dir} --output_dir ${saved_file} --eval_match_top_n 1
