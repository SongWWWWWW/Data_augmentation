# get data and parse and combined 
# combine sample
import json
import argparse
import re
from sympy import EX
from prompt import get_prompt, get_task_name, get_task_description
from response import get_batch_response
import csv
import os
import pandas as pd
from find_path import find_path
def format_prompt(data,prompt, task,description,spurious_num,generate_num):
    # NOTE
    # model's label { "FAVOR": 0  , "NONE": 1 , "AGAINST":  2}
    # the data is from model's inference
    # but that is needed to transform to { "FAVOR": 1, "NONE": 2, "AGAINST": 0}
    transform_dict = {
        0: 'FAVOR',
        1: 'NONE',
        2: 'AGAINST'
    }
    prompts = []
    for d in data:
        d_ = {
            "input_text": d["input_text"],
            "target": d["target"],
            "predicted_label": transform_dict[d["predicted_label"]],
            "true_label": transform_dict[d["true_label"]]
        }
        prompts.append(prompt.format(task_name = task, task_description = description, spurious_num = spurious_num, generate_num = generate_num, data = d_))
    return prompts
def get_raw_response(args):
    prompt = get_prompt(args.prompt)
    model = args.model
    task = args.task
    description = args.task_description
    generate_num = args.generate_num
    spurious_num = args.spurious_num
    with open(args.dev_wrong_path,"r") as f:
        data = json.load(f)
    prompts = format_prompt(data,prompt,task,description,spurious_num,generate_num)
    response = get_batch_response(model,prompts)
    return response
    
def parse_args():
    parser = argparse.ArgumentParser(description="Data processing script.")
    
    # 添加参数
    parser.add_argument('--dev_wrong_path', type=str, required=True, help="Path to the development wrong data.")
    parser.add_argument('--model', type=str, required=True, help="Model name.")
    parser.add_argument('--prompt', type=str, required=True, help="Prompt for model generation.")
    parser.add_argument('--task', type=str, required=True, help="task name")
    parser.add_argument('--task_description', type=str, required=True, help="task description")
    parser.add_argument('--generate_num', type=int, required=True, help="Number of generations.")
    parser.add_argument('--spurious_num', type=int, required=True, help="Number of spurious data points.")
    parser.add_argument('--split_dev_rate', type=float, default=0.3, help="the rate of split to dev(response), train-0.7,dev-0.3,split_dev_rate-0.3 ")
    parser.add_argument('--raw_answer_save_path', type=str, required=True, help="Path to save raw answers.")
    parser.add_argument('--raw_dev_path', type=str, required=True, help="Path to the previous dev data.")
    parser.add_argument('--raw_train_path', type=str, required=True, help="Path to the previous train data.")
    parser.add_argument('--sample_rate', type=float, required=True,help="sample generated data that model can't classfie")
    parser.add_argument('--model_path', type=str, help="if 1 > sample_rate > 0, model_path required, that is the model of dev_wrong_path ")
    parser.add_argument('--trial_name', type=str, help="be used to sign this trial")
    return parser.parse_args()
def save(data, path):
    try:
        with open(path,"w") as f:
            json.dump(data,f,indent=4)
        return True
    except Exception as e:
        print(e)
        return False
def combine_data(path1,path2,save_path):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    # 按行合并，自动对齐列
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # 保存结果
    combined_df.to_csv(save_path, index=False)
def model_output_transform_to_csv(data,path):
    # data
    transfer = {
        0: "FAVOR",
        1: "NONE",
        2: "AGAINST",
        'FAVOR': "FAVOR",
        'NONE': "NONE",
        'AGAINST': "AGAINST"
        
    }
    write_data = []

    for j in data:
            try:
                s = []
                s.append(j["text"])
                s.append(j["target"])
                s.append(transfer[j["ground_truth"].strip(" ")])
                s.append(1)
                s.append("invalid")
                write_data.append(s)
            except Exception as e:
                # print(j)
                pass
    csv_filename = path
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Tweet', 'Target 1', 'Stance 1', 'seen?', 'GT Target'])  # 写入标题行
        writer.writerows(write_data)  # 写入数据行
        
        
def transform_to_csv(data,path):
    # data
    transfer = {
        '0': "AGAINST",
        '1': "FAVOR",
        '2': "NONE",
        'FAVOR': "FAVOR",
        'NONE': "NONE",
        'AGAINST': "AGAINST"
        
    }
    write_data = []

    for j in data:
            try:
                s = []
                s.append(j["text"])
                s.append(j["target"])
                s.append(transfer[j["ground_truth"].strip(" ")])
                s.append(1)
                s.append("invalid")
                write_data.append(s)
            except Exception as e:
                # print(j)
                pass
    csv_filename = path
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Tweet', 'Target 1', 'Stance 1', 'seen?', 'GT Target'])  # 写入标题行
        writer.writerows(write_data)  # 写入数据行

def parse(answer:str,args):
    pattern = r"<text>(.*?)</text>\s*<target>(.*?)</target>\s*<ground_truth>(.*?)</ground_truth>"
    matches = re.findall(pattern, answer)
    ans = []
    for match in matches:
        ans.append({
            "text":match[0],
            "target":match[1],
            "ground_truth":match[2]
        })
    if len(ans) != args.spurious_num*args.generate_num:
        return []
    return ans

# def sample(args):
#     import json
#     path = "error file"
#     with open(path,"r") as f:
#         data = json.load(f)
#     index = []
#     for i in range(527):
#         index.append([0]*3)
#         for x in data:
#             if x["index"] >= i*60 and x["index"] < (i+1)*60:
#                 index[i][(x["index"]-i*60)//20] += 1
#     ans = []
#     for index_i,i in enumerate(index):
#         for index_j,j in enumerate(i):
#             if j == 20:
#                 ans.append(index_i*60+index_j*20)
#     print(len(ans))
#     ans
#     with open("/home/ubuntu/wcc/now-task/data/train/spurious_1_60_parse.json","r") as f:
#         parse = json.load(f)
#     final_data = []
#     for i in ans:
#         final_data += parse[i:i+20]

#     with open("/home/ubuntu/wcc/now-task/data/train/train_spurious_1_60_parse_100%.json","w") as f:
#         json.dump(final_data,f,indent=4)
        
        
args = parse_args()
print(args)
# NOTE please change the code followed before figuring out the content of ".rstrip "
raw_response_path = args.raw_answer_save_path
parse_response_path = raw_response_path.rstrip(".json") + "_parse.json"
csv_parse_response_path = parse_response_path.rstrip(".json") + ".csv"
parse_response_dev_path = parse_response_path.rstrip(".json") + "_dev.json"
csv_parse_response_dev_path = parse_response_dev_path.rstrip(".json") + ".csv"

combined_train_path =  csv_parse_response_path.rstrip(".csv") + "_combined.csv"
combined_dev_path = csv_parse_response_dev_path.rstrip("v").rstrip(".cs") + "_combined.csv"
# generate data

if not os.path.exists(raw_response_path):
    raw_response = get_raw_response(args)
    save(raw_response,raw_response_path)
    print(f"Get Data Completely！ Save to [{raw_response_path}]")
else: 
    with open(raw_response_path,"r") as f:
        raw_response = json.load(f)
    print(f"Data is existed, path: {raw_response_path}")

parse_response = []  
for r in raw_response:
    parse_response += parse(r,args)
print(f"Parse Data Completely！ length = {len(parse_response)}")

log_delete = []
groups = [parse_response[i:i + args.generate_num] for i in range(0, len(parse_response), args.generate_num)]
filtered_groups = [
    group for group in groups 
    if all(item["ground_truth"] in ["0","1","2","AGAINST","NONE","FAVOR"] for item in group)
]
parse_response = [item for group in filtered_groups for item in group]
# process data eg: sample transform 
# sample_rate: Setting sample_rate equals 0.8, if the large model generate 10 instances per spurious pattern, and the pre model can't classify
# 8 or 8+, and then this 10 instances will be added to train or dev dataset.
if args.sample_rate <= 1.0:
    transform_to_csv(parse_response,csv_parse_response_path)
    # TEMP
    if not os.path.exists(os.path.join(args.model_path,"logs",f"log_sample_{args.trial_name}_eval_log.json")):
        os.system(
            f"python eval.py "
            f"--model_root_path '{args.model_path}' "
            f"--test_path '{csv_parse_response_path}' "
            f"--output_error_question False "
            f"--save_log True "
            f"--sign sample_{args.trial_name} "
        )
    # NOTE the data that model generate shoule be transform to normal label
    best_checkpoint = find_path(args.model_path)
    error_path = os.path.join(best_checkpoint, f"log_sample_{args.trial_name}_wrong_test_data.json")
    
    with open(error_path,"r") as f:
        error_data = json.load(f)
    print(f"Model path: {args.model_path}, error data length: {len(error_data)}")
    index_error = []
    error_target_position = []
    for index, i in enumerate(parse_response):
        if i["ground_truth"] not in ["0","1","2","AGAINST","NONE","FAVOR"]:
            error_target_position.append(index)
    for i in range(int(len(parse_response)/(args.spurious_num*args.generate_num)) + 1):
        index_error.append([0]*args.spurious_num)
        for x in error_data:
            for j in error_target_position:
                # NOTE here maybe add again
                if x["index"] > j:
                    x["index"] += 1
            if x["index"] >= i*args.spurious_num*args.generate_num and x["index"] < (i+1)*args.spurious_num*args.generate_num:
                index_error[i][(x["index"]-i*args.spurious_num*args.generate_num)//args.generate_num] += 1 # count
    ans = []
    for index_i,i in enumerate(index_error):
        for index_j,j in enumerate(i):
            if j >= args.generate_num*args.sample_rate:
                ans.append(index_i*args.spurious_num*args.generate_num + index_j*args.generate_num)
    train = []
    dev = []
    for i in ans:
        print(i)
        dev += parse_response[i:i+int(args.split_dev_rate*args.generate_num)]
        train += parse_response[i+int(args.split_dev_rate*args.generate_num):i+1*args.generate_num]
    
# combined

save(train,parse_response_path)
print(f"Split Data Completely！ Train Data Save to [{parse_response_path}]! length = {len(train)}")

save(dev,parse_response_dev_path)
print(f"Split Data Completely！ Dev Data Save to [{parse_response_dev_path}]! length = {len(dev)}")

# model output label ?
model_output_transform_to_csv(train, csv_parse_response_path )
model_output_transform_to_csv(dev, csv_parse_response_dev_path)

combine_data(args.raw_train_path,csv_parse_response_path,combined_train_path)
combine_data(args.raw_dev_path,csv_parse_response_dev_path,combined_dev_path)
print(f"Combine Data Completely！ Train Data Save to [{combined_train_path}]")
print(f"Combine Data Completely！ Dev Data Save to [{combined_dev_path}]")
