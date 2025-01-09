#!/bin/bash

python trainer.py \
  --num_labels 3 \
  --batch_size 64 \
  --epochs 10 \
  --gpu \
  --train_path "../vast/raw_train_all_oneall.csv" \
  --val_path "../vast/raw_val_all_oneall.csv" \
  --save_path "./results/baseline"


python eval.py\
    --model_root_path './results/baseline'\
    --test_path '../vast/raw_test_all_onecol.csv'\
    --output_error_question False\
    --save_log True\
    --sign "test"

python eval.py\
    --model_root_path './results/baseline'\
    --test_path '../vast/raw_dev_all_onecol.csv'\
    --output_error_question False\
    --save_log True\
    --sign "dev"