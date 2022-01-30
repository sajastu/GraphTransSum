#!/usr/bin/env bash

############### Normal- experiments Longsumm #################

MODE_LINES=format_to_lines_simple
MODE_BERT=format_to_bert
export BASE_DIR=/disk1/sajad/datasets/sci/arxivL/
export RAW_PATH=$BASE_DIR/single-files
export SAVE_JSON=$BASE_DIR/jsons/intro-normal-BertScore-full/
#export BERT_DIR=$BASE_DIR/bert-files/arXivL-2048-absNegReferece-tgtIncluded-reduced/
export BERT_DIR=$BASE_DIR/bert-files/arxivl-2048-graph-tgt/


#echo "Starting to write aggregated json files..."
#echo "-----------------"
#for SET in train val test
#do
#    python3 preprocess.py -mode $MODE_LINES \
#                        -save_path $SAVE_JSON  \
#                        -n_cpus 12 \
#                        -keep_sect_num \
#                        -shard_size 1999 \
#                        -log_file ../logs/preprocess.log \
#                        -raw_path $RAW_PATH/$SET/ \
#                        -dataset $SET
#done



echo "-----------------"
echo "Now starting to write torch files..."
echo "-----------------"

for SET in train
do
  python3 preprocess.py -mode $MODE_BERT \
                        -bart \
                        -model_name longformer \
                        -raw_path $SAVE_JSON/ \
                        -save_path $BERT_DIR/ \
                        -n_cpus 12 \
                        -log_file ../logs/preprocess.log \
                        -lower \
#                        -dataset $SET \
#                        -sent_numbers_file save_lists/lsum-$SET-longformer-multi50-aftersdu-top-sents.p
done

#python3 spliting_bertfiles.py -pt_dirs_src $BERT_DIR
#
#echo '------------------------'
#echo 'ORACLE SCORE FOR DEV'
##python calculate_oracle_from_bertfiles.py -pt_dirs_src $BERT_DIR -set val
#
##echo '------------------------'
##echo 'ORACLE SCORE FOR TEST'
##python calculate_oracle_from_bertfiles.py -pt_dirs_src $BERT_DIR -set test
#
#export BASE_DIR=/disk1/sajad/datasets/sci/arxivL/
#export RAW_PATH=$BASE_DIR/intro_summary/splits-normal2-15/
#export SAVE_JSON=$BASE_DIR/jsons/intro-normal-BertScore-full/
#export BERT_DIR=$BASE_DIR/bert-files/arXivL-2048-absNegRandom-tgtIncluded/
