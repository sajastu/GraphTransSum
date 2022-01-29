#!/usr/bin/env bash

################################################################################
##### CHECKPOINTS –to train from #######
################################################################################
#CHECKPOINT=/disk1/sajad/sci-trained-models/presum/arxiv-first-phase/model_step_30000.pt

export SERVER=$(hostname)

################################################################################
##### Global Params #######
################################################################################
export DATASET=arxivL
export N_SENTS=10
export TRAINING_STEPS=100000
export VAL_INTERVAL=1000

#METHOD=ABSsummary-RG-truncated
#METHOD=ARXIVL-tgtInformerd-Margin2
export CUDA_VISIBLE_DEVICES=1
export WANDB_START_METHOD=thread

MAX_POS=2500
MAX_POS_INTRO=1024

ROW_NUMBER=110
export GD_CELLS_RG_VAL=D$ROW_NUMBER:F$ROW_NUMBER
export GD_CELLS_RG_TEST=H$ROW_NUMBER:J$ROW_NUMBER

export GD_CELLS_RECALL_VAL=G$ROW_NUMBER
export GD_CELLS_RECALL_TEST=K$ROW_NUMBER

export GD_CELLS_STEP=Q$ROW_NUMBER


BSZ=1

################################################################################
##### Data #######
################################################################################

#DATA_PATH=/disk1/sajad/datasets/sci/$DATASET/bert-files/$MAX_POS-segmented-intro$MAX_POS_INTRO-$N_SENTS-introConc-test/
#BERT_DIR=/disk1/sajad/datasets/sci/arxivL/intro_summary/bert-files/arXivL-introSumm-2048-wRg-truncated/
################################################################################
##### MODEL #######
################################################################################
MODEL_DB=`echo $DATASET | tr 'a-z' 'A-Z'`
METHOD=BERTSUM-graphCompar-reduced
#METHOD=BERTSUM-reduced
export MODEL_PATH=/disk1/sajad/sci-trained-models/presum/$MODEL_DB-$MAX_POS-$METHOD-$SERVER/

################################################################################
##### TRAINING SCRIPT #######
################################################################################


LOG_DIR=../logs/$(echo $MODEL_PATH | cut -d \/ -f 6).log
mkdir -p ../results/$(echo $MODEL_PATH | cut -d \/ -f 6)
RESULT_PATH_TEST=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/

python train.py -task ext \
                -mode train \
                -intro_cls \
                -model_name longformer \
                -val_pred_len $N_SENTS \
                -bert_data_path $BERT_DIR \
                -ext_dropout 0.1 \
                -model_path $MODEL_PATH \
                -lr 2e-3 \
                -visible_gpus $CUDA_VISIBLE_DEVICES \
                -report_every 50 \
                -log_file $LOG_DIR \
                -val_interval $VAL_INTERVAL \
                -save_checkpoint_steps 300000 \
                -batch_size $BSZ \
                -test_batch_size 1 \
                -max_length 600 \
                -train_steps $TRAINING_STEPS \
                -alpha 0.95 \
                -use_interval true \
                -warmup_steps 4000 \
                -max_pos $MAX_POS \
                -gd_cells_rg $GD_CELLS_RG_VAL \
                -gd_cell_step $GD_CELLS_STEP \
                -max_pos_intro $MAX_POS_INTRO\
                -result_path_test $RESULT_PATH_TEST \
                -accum_count 2 \
                -finetune_bert
