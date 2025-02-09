# /bin/bash

python test1.py \
    --task_name test_6k \
    --train_path "../vast/raw_train_all_onecol.csv" \
    --val_path "../vast/raw_val_all_onecol.csv" \
    --test_path "../vast/raw_test_all_onecol.csv" \
    --train_split 6000
python test1.py \
    --task_name test_7k \
    --train_path "../vast/raw_train_all_onecol.csv" \
    --val_path "../vast/raw_val_all_onecol.csv" \
    --test_path "../vast/raw_test_all_onecol.csv" \
    --train_split 7000
python test1.py \
    --task_name test_8k \
    --train_path "../vast/raw_train_all_onecol.csv" \
    --val_path "../vast/raw_val_all_onecol.csv" \
    --test_path "../vast/raw_test_all_onecol.csv" \
    --train_split 8000
python test1.py \
    --task_name test_9k \
    --train_path "../vast/raw_train_all_onecol.csv" \
    --val_path "../vast/raw_val_all_onecol.csv" \
    --test_path "../vast/raw_test_all_onecol.csv" \
    --train_split 9000

python test1.py \
    --task_name test_9k \
    --train_path "../vast/raw_train_all_onecol.csv" \
    --val_path "../vast/raw_val_all_onecol.csv" \
    --test_path "../vast/raw_test_all_onecol.csv" \
    --train_split 9000
