import os
from find_path import find_path
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Parse arguments for task configuration.")
    parser.add_argument("--task_name", type=str, required=True, help="The name of the task to execute.")
    parser.add_argument("--sample_rate", type=float, default=0.0, help="The sampling rate for the task. Default is 0.0.")
    parser.add_argument("--data_generate_strategy", type=str, default=None,help="the strategy of generating data")
    parser.add_argument("--generate_num",type=int,default=10,help="the number of generated instances")
    # 解析命令行参数
    args = parser.parse_args()
    return args
args = parse_args()
task_name = args.task_name
sample_rate = args.sample_rate
data_generate_strategy = str(args.data_generate_strategy)
# task 配置
# task_name = "prompt2_3.1_7_3_0.8"
# 参数配置
num_iterations = 4  # 总共迭代轮数
base_data_path = f"../data/train/{task_name}"
os.makedirs(base_data_path, exist_ok=True)
base_results_path = f"../results/{task_name}_iter"
base_raw_train_path = "../vast/raw_train_all_onecol.csv"
base_raw_val_path = "../vast/raw_val_all_onecol.csv"
base_raw_test_path = "../vast/raw_test_all_onecol.csv"

# 模型和训练配置
model_name = "meta.llama3-1-70b-instruct-v1:0"
prompt = "prompt2"
task = "task1"
description = "description1"

split_dev_rate = 0.3
# sample_rate = 0.8

generate_num = args.generate_num
spurious_num = 3
batch_size = 64
epochs = 10
num_labels = 3
use_gpu = True

# 开始迭代
prev_train_path = base_raw_train_path
prev_dev_path = base_raw_val_path
dev_wrong_path = "./results/baseline/checkpoint-318/log_dev_wrong_test_data.json"
# dev_wrong_path = "../results/baseline/checkpoint-216/log_dev_wrong_test_data.json"
# pre_model_path = "../results/baseline"
pre_model_path = "./results/baseline"
# dev_wrong_path = os.path.join(find_path(pre_model_path),"log_dev_wrong_test_data.json")
for i in range(1, num_iterations + 1):
    iter_name = f"iter{i}"
    current_results_path = f"{base_results_path}{i}/"
    current_raw_response_path = f"{base_data_path}/{iter_name}_raw_response.json"
    current_train_path = f"{base_data_path}/{iter_name}_raw_response_parse_combined.csv"
    current_dev_path = f"{base_data_path}/{iter_name}_raw_response_parse_dev_combined.csv"

    # 数据处理
    # TODO

    os.system(
        f"python data_process.py "
        f"--dev_wrong_path '{dev_wrong_path}' " # change √
        f"--model '{model_name}' "
        f"--prompt {prompt} "
        f"--task {task} "
        f"--task_description {description} "
        f"--generate_num {generate_num} "
        f"--spurious_num {spurious_num} "
        f"--split_dev_rate {split_dev_rate} "
        f"--raw_answer_save_path '{current_raw_response_path}' " # change √
        f"--raw_dev_path '{prev_dev_path}' " # change √
        f"--raw_train_path '{prev_train_path}' " # change √
        f"--sample_rate {sample_rate} "
        f"--model_path {pre_model_path} " # change √
        f"--trial_name {task_name} "
        f"--data_generate_strategy {data_generate_strategy} "
    )

    # 模型训练
    os.system(
        f"python trainer.py "
        f"--num_labels {num_labels} "
        f"--batch_size {batch_size} "
        f"--epochs {epochs} "
        f"--gpu "
        f"--train_path '{current_train_path}' " # change √
        f"--val_path '{current_dev_path}' " # change √
        f"--save_path '{current_results_path}' " # change √
    )

    # 测试评估
    for sign, test_path in [("test", base_raw_test_path), ("dev", current_dev_path),("raw_dev", base_raw_val_path)]:
        os.system(
            f"python eval.py "
            f"--model_root_path '{current_results_path}' "
            f"--test_path '{test_path}' "
            f"--output_error_question False "
            f"--save_log True "
            f"--sign '{sign}' "
        )

    # 更新路径
    prev_train_path = current_train_path
    prev_dev_path = current_dev_path
    dev_wrong_path = os.path.join(find_path(current_results_path),"log_dev_wrong_test_data.json")
    pre_model_path = current_results_path