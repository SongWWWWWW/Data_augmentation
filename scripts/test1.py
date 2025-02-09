import os
import argparse
import random
import pandas as pd
from find_path import find_path
from data_generate_strategy import get_strategy
from prompt import get_task_description, get_task_name
from data_process import transform_to_csv,combine_data
def split_csv(input_file, n, output_selected, output_remaining):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 确保 n 不超过数据总行数
    n = min(n, len(df))

    # 随机挑选 n 行数据
    selected_indices = random.sample(range(len(df)), n)
    selected_df = df.iloc[selected_indices]

    # 获取剩余数据
    remaining_df = df.drop(selected_indices)

    # 保存两个 CSV 文件
    selected_df.to_csv(output_selected, index=False)
    remaining_df.to_csv(output_remaining, index=False)
    
def parse_args():
    parser = argparse.ArgumentParser(description="1w + 4k train data, 4k to product new data")
    parser.add_argument("--task_name",type=str,help="task name")
    parser.add_argument("--train_path",type=str,help="train data path")
    parser.add_argument("--val_path",type=str,help="dev data path")
    parser.add_argument("--test_path",type=str,help="test data path")
    parser.add_argument("--train_split",type=int,help="the num of spliting data ")
    # 解析命令行参数
    args = parser.parse_args()
    return args
spurious_num = 3
batch_size = 64
epochs = 10
num_labels = 3
use_gpu = True
seed = 42
random.seed(seed)
args = parse_args()
data_path = f"../data/train/{args.task_name}"
if not os.path.exists(data_path):
    os.mkdir(data_path)
selected_data_path = os.path.join(data_path,"selected_path.csv")
remained_data_path = os.path.join(data_path,"remained_path.csv")
split_csv(args.train_path,args.train_split,selected_data_path,remained_data_path)
model_root_path = f"../results/{args.task_name}_baseline"
# train
if not os.path.exists(model_root_path):
    os.system(
        f"python trainer.py "
        f"--num_labels {num_labels} "
        f"--batch_size {batch_size} "
        f"--epochs {epochs} "
        f"--gpu "
        f"--train_path '{remained_data_path}' " # change √
        f"--val_path '{args.val_path}' " # change √
        f"--save_path '{model_root_path}' " # change √
    )
# predict
os.system(
            f"python eval.py "
            f"--model_root_path '{model_root_path}' "
            f"--test_path '{selected_data_path}' "   # <- 4k data
            f"--output_error_question False "
            f"--save_log True "
            f"--sign '{args.task_name}' "
        )
os.system(
            f"python eval.py "
            f"--model_root_path '{model_root_path}' "
            f"--test_path '{args.test_path}' "   
            f"--output_error_question False "
            f"--save_log True "
            f"--sign 'test' "
        )
os.system(
            f"python eval.py "
            f"--model_root_path '{model_root_path}' "
            f"--test_path '{args.val_path}' "   
            f"--output_error_question False "
            f"--save_log True "
            f"--sign 'dev' "
        )
model_chckp = find_path(model_root_path) 
wrong_data_path = os.path.join(model_chckp,f"log_{args.task_name}_wrong_test_data.json")
# generate spurious pattern
# generate new data
model_generate_strategy = get_strategy("strategy1")

strategy_args = {
    "dev_wrong_path": wrong_data_path, 
    "raw_answer_save_path": os.path.join(data_path,"raw_response.json"), 
    "task": "task1",
    "task_description":"description1",
    "spurious_num":spurious_num,
    "generate_num":5,
    
    }
strategy_args = argparse.Namespace(**strategy_args)
new_data = model_generate_strategy(strategy_args)
transform_to_csv(new_data,os.path.join(data_path,"augmentation.csv"))
combine_data(remained_data_path,os.path.join(data_path,"augmentation.csv"),os.path.join(data_path,"new_train.csv"))
# train
os.system(
    f"python trainer.py "
    f"--num_labels {num_labels} "
    f"--batch_size {batch_size} "
    f"--epochs {epochs} "
    f"--gpu "
    f"--train_path '{os.path.join(data_path,'new_train.csv')}' " # change √
    f"--val_path '{args.val_path}' " # change √
    f"--save_path '{f'{model_root_path}_new_train'}' " # change √
)
# test
os.system(
            f"python eval.py "
            f"--model_root_path '{f'{model_root_path}_new_train'}' "
            f"--test_path '{args.test_path}' "   # <- 4k data
            f"--output_error_question False "
            f"--save_log True "
            f"--sign 'test' "
        )
os.system(
            f"python eval.py "
            f"--model_root_path '{f'{model_root_path}_new_train'}' "
            f"--test_path '{args.val_path}' "   # <- 4k data
            f"--output_error_question False "
            f"--save_log True "
            f"--sign 'dev' "
        )