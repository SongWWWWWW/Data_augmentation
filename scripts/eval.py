from transformers import Trainer, TrainingArguments
from roberta_mlp import RoBERTa_MLP
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, RobertaTokenizer
from trainer import prepare_data
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import json
from parse_args import eval_parse_args
import os

def compute_metrics(p):
    predictions, labels = p
    # 将预测结果转换为类别索引
    preds = predictions.argmax(axis=1)
    
    # 将独热编码标签转换为类别索引
    labels = labels.argmax(axis=1)

    return {"f1": f1_score(labels, preds, average="macro")}  # 返回宏平均 F1 分数

def get_model_path(path:str):
    paths = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for dir_name in dirs:
            if dir_name == "logs":
                continue
            dir_path = os.path.join(root, dir_name)
            # print("Folder:", dir_path)            
            paths.append(dir_path)
    return paths

args = eval_parse_args()
print(args)
model_path = sorted(get_model_path(args.model_root_path), key=lambda x: int(x.split('-')[-1]))
print(model_path)
# model_path = get_model_path(args.model_root_path)  
result = []
wrong_predictions= []
log = {}
log["TestSet_path"] = args.test_path
for i in model_path:
    wrong_predictions = []
    path = f"{i}/model.safetensors"  # 或 "{i}/pytorch_model.bin"
    model = RoBERTa_MLP(num_labels=3)

    # 加载 .safetensors 权重
    state_dict = load_file(path)

    # 将加载的权重加载到模型中
    model.load_state_dict(state_dict)


    # path = """../vast/raw_{}_all_onecol.csv"""
    test_dataset = prepare_data(args.test_path)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    # # 假设你已经有一个 Trainer 实例
    # training_args = TrainingArguments(
    #     output_dir='./r',
    #     evaluation_strategy="epoch",  # 或者 "steps" 根据你的需要
    #     per_device_eval_batch_size=16,
    # )

    trainer = Trainer(
        model=model,  # 你的模型
        args=None,
        eval_dataset=test_dataset,  # 你的测试集
        compute_metrics=compute_metrics
    )
    print(f"\033[31m Test_checkpoint: {i}\033[0m")
    predictions, label_ids, metrics = trainer.predict(test_dataset)
    preds = predictions.argmax(axis=1)  # 预测标签索引
    labels = label_ids.argmax(axis=1)  # 真实标签索引

    for idx, (pred, true) in enumerate(zip(preds, labels)):
        if pred != true:
            wrong_predictions.append({
                "index": idx,
                "true_label": true,
                "predicted_label": pred,
                "input_text": test_dataset[idx]["text"],  # 假设你的数据集中有 "text" 字段
                "target": test_dataset[idx]["target"]
            })
        else:
            # print(1)
            pass

    # 输出做错的题目
    if args.output_error_question is True:
        print("模型做错的题目：")
        for sample in wrong_predictions:
            print(f"索引: {sample['index']}, 真实标签: {sample['true_label']}, 预测标签: {sample['predicted_label']}, target: {sample['target']}")
            
            print(f"文本: {sample['input_text']}")
            print("-" * 50)

    wrong_predictions = [
        {key: (int(value) if isinstance(value, np.int64) else value) for key, value in sample.items()}
        for sample in wrong_predictions
    ]
    if args.save_log:
        with open(os.path.join(i,f"log_{args.sign}_wrong_test_data.json"),"w") as f:
            json.dump(wrong_predictions,f,indent=4)
    print("总错误数：", len(wrong_predictions))
    print("错误率: ", len(wrong_predictions)/len(test_dataset)*100,"%")
    # print(metrics)
    print("F1 score: ", metrics["test_f1"])
    log[i] = {
        "Num of questions": len(test_dataset),
        "Error rate": str(len(wrong_predictions)/len(test_dataset)*100)+"%",
        "F1 score": metrics["test_f1"]
    }
    result.append(metrics["test_f1"])  # 将 F1 分数添加到结果列表
if args.save_log:
    with open(os.path.join(args.model_root_path,"logs",f"log_{args.sign}_eval_log.json"),"w") as f:
        json.dump(log,f,indent=4)
plt.figure(figsize=(10, 5))  # 设置图形的大小
plt.plot(result, marker='o', linestyle='-', color='b')  # 使用圆点标记每个数据点，并设置线条样式和颜色

# 添加标题和标签
plt.title('test set f1 score / epoch')
plt.xlabel('epoch')
plt.ylabel('f1 score')
if args.save_log:
    plt.savefig(os.path.join(args.model_root_path, "logs",f"log_{args.sign}_testset_f1score.png"))  
# 显示图形
plt.show()
