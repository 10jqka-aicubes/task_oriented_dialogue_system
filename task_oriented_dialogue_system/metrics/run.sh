#!/usr/bin/bash

basepath=$(cd `dirname $0`; pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath

reference_file_path=$GROUNDTRUTH_FILE_DIR'/groundtruth'
predict_file_path=$PREDICT_RESULT_FILE_DIR'/predict'
result_file=$RESULT_JSON_FILE
bert_dir='/read-only/common/pretrain_model/transformers/bert-base-chinese/'

python eval_process.py --reference_file_path ${reference_file_path} --predict_file_path ${predict_file_path} --bert_dir ${bert_dir} --result_file ${result_file}