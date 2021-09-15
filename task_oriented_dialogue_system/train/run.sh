#!/usr/bin/bash

basepath=$(cd `dirname $0`; pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath


model=baseline_lmcl
version=1.0
batch=8
lr=1e-4
seed=1111
num_train_epochs=300
patience=5

bert_dir='/read-only/common/pretrain_model/transformers/bert-base-chinese/'
python train.py --do_train --model ${model} --data_dir ${TRAIN_FILE_DIR} --bert_dir ${bert_dir} --dialog_encoder_file ${bert_dir} --output_dir ${SAVE_MODEL_DIR} --num_train_epochs=${num_train_epochs} --patience=${patience}