from roberta_mlp import RoBERTa_MLP
from data_loader import TextDataset
import argparse
import torch
from torch import nn
from transformers import RobertaModel, RobertaTokenizer, TrainerCallback
from parse_args import parse_args
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score
from data_loader import PrepareData
import matplotlib.pyplot as plt
import os
import json
import math
def compute_metrics(p):
    predictions, labels = p
    # 将预测结果转换为类别索引
    preds = predictions.argmax(axis=1)
    
    # 将独热编码标签转换为类别索引
    labels = labels.argmax(axis=1)
    # print("preds: ",preds,"\nlabels: ",labels)

    return {"f1": f1_score(labels, preds, average="macro")}  # 返回宏平均 F1 分数

def prepare_data(path):
    data = PrepareData(path)
    sample_texts = data.texts
    sample_target = data.target
    labels = data.tensor_labels

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    encodings = tokenizer(sample_texts, truncation=True, add_special_tokens=True, padding=True, return_tensors="pt")
    encodings_target = tokenizer(sample_target,truncation=True, add_special_tokens=False, padding=True,return_tensors="pt")
    encodings_ids = torch.cat((encodings["input_ids"],encodings_target["input_ids"]),dim=1)
    # print("Type of encodings['attention_mask']:", type(encodings["attention_mask"]))
    # print("Contents of encodings['attention_mask']:", encodings["attention_mask"])

    # print("Shape of encodings['attention_mask']:", encodings["attention_mask"].shape)
    # print("Shape of encodings_target['attention_mask']:", encodings["attention_mask"].shape)

    attention_mask = torch.cat((encodings["attention_mask"], encodings_target["attention_mask"]), dim=1)

    # Create dataloaders
    train_dataset = TextDataset(encodings_ids, labels, attention_mask,sample_texts,sample_target)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataset

class MetricsRecorderCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.eval_losses = []
        self.eval_f1_scores = []
        self.log_epoch = []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # print(logs)
            # if "loss" in logs:
            #     self.train_losses.append(logs["loss"])
            # print(f"Logs: {logs}")
            # if math.floor(logs["epoch"]) == logs["epoch"]:
            #     print(f"Epoch: {logs['epoch']}")
                # self.train_losses.append(logs["loss"]) 
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
            if "eval_f1" in logs:
                self.eval_f1_scores.append(logs["eval_f1"])
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                # print("loss: ",logs["loss"])
            if "epoch" in logs and len(self.train_losses) > len(self.log_epoch) :
                self.log_epoch.append(logs["epoch"])
                # print("epoch: ", logs["epoch"])         

if __name__ == "__main__":

    metrics_callback = MetricsRecorderCallback()
    args = parse_args()

    model = RoBERTa_MLP(num_labels=args.num_labels)
    # path_train = "../data/combine_train_generate_data.csv"
    # path = """../vast/raw_{}_all_onecol.csv"""
    train_dataset = prepare_data(args.train_path)
    eval_dataset = prepare_data(args.val_path)
    log_path = os.path.join(args.save_path,"logs")
    if not os.path.exists(log_path):
    # 如果文件夹不存在，则使用os.makedirs()创建文件夹
        os.makedirs(log_path)
        print(f"文件夹'{log_path}'已创建。")
    else:
        print(f"文件夹'{log_path}'已存在。")

    # loss_fn = nn.CrossEntropyLoss()




    training_args = TrainingArguments(
        output_dir=args.save_path,          # 输出文件夹
        evaluation_strategy="epoch",     # 每个 epoch 后评估一次
        save_strategy="epoch",           # 每个 epoch 后保存模型
        learning_rate=2e-5,              # 学习率
        per_device_train_batch_size=args.batch_size,  # 训练时每个设备的批次大小
        num_train_epochs=args.epochs,              # 训练轮数
        lr_scheduler_type="linear",      # 选择线性学习率调度器
        warmup_steps=10,                # 设置预热步数
        logging_steps=1,                    # 每一步记录日志
        logging_dir=log_path            # 指定日志文件夹
    )

    trainer = Trainer(
        model=model,                     # 你的模型
        args=training_args,              # 训练参数
        train_dataset=train_dataset,     # 训练数据
        eval_dataset=eval_dataset,       # 开发集
        compute_metrics=compute_metrics, # 评估指标
        callbacks=[metrics_callback]
    )

    trainer.train()
    # trainer.save_model(args.save_path)
    log_train = {
        "train_losses": metrics_callback.train_losses,
        "eval_losses": metrics_callback.eval_losses,
        "eval_f1_scores": metrics_callback.eval_f1_scores
    }
    with open(os.path.join(log_path,"train_process_log.json"),"w") as f:
        json.dump(log_train,f,indent=4)
    plt.figure(figsize=(12, 6))

    # 绘制训练损失曲线
    plt.plot(metrics_callback.log_epoch, metrics_callback.train_losses, label='Training Loss', color='blue')
    # plt.plot(metrics_callback.train_losses, label='Training Loss', color='blue')
    # 绘制验证损失曲线
    plt.plot(metrics_callback.eval_losses, label='Validation Loss', color='orange')
    # 绘制验证 F1 曲线
    plt.plot(metrics_callback.eval_f1_scores, label='Validation F1 Score', color='green')

    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("Training and Validation Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_path,"training_validation_metrics.png"), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"图像已保存到：{os.path.join(log_path,'training_validation_metrics.png')}")