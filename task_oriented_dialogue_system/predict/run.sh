#!/usr/bin/bash

basepath=$(cd `dirname $0`; pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath

mkdir -p $PREDICT_RESULT_FILE_DIR
test_data=$PREDICT_FILE_DIR'/test.json'
test_anwser_candidates=$PREDICT_FILE_DIR'/answer_candidates.json'
predict_file_path=$PREDICT_RESULT_FILE_DIR'/predict'
bert_dir='/read-only/common/pretrain_model/transformers/bert-base-chinese/'
saved_model_dir=$SAVE_MODEL_DIR
python predict_process.py --test_data=${test_data} --test_anwser_candidates=${test_anwser_candidates} --bert_dir=${bert_dir} --dialog_encoder_dir=${bert_dir} --saved_model_dir=${saved_model_dir} --predict_file_path=${predict_file_path}