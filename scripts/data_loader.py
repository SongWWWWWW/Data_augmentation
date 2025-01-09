import torch
import pandas as pd
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, attention_mask,text,target):
        self.encodings = encodings
        self.labels = labels
        self.attention_mask = attention_mask
        self.text = text
        self.target = target
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 将 `input_ids` 和 `attention_mask` 添加到 `item`
        item = {
            'text': self.text[idx],
            'target': self.target[idx],
            'input_ids': self.encodings[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx].clone().detach()
        }
        return item

class PrepareData:
     
    def __init__(self,path:str):
        self.data = pd.read_csv(path)
        self.texts = self.data["Tweet"].to_list()
        self.target = self.data["Target 1"].to_list()
        self.labels = self.data["Stance 1"].to_list()        
        self.transform_label()
    def transform_label(self):
        self.tensor_labels = []
        for i in self.labels:
            # 创建一个长度为3的零向量
            one_hot = torch.zeros(3)  # 假设有3个类别
            match i:
                case "FAVOR":
                    one_hot[0] = 1  # Favor 类别对应位置为 1
                case "NONE":
                    one_hot[1] = 1  # None 类别对应位置为 1
                case "AGAINST":
                    one_hot[2] = 1  # Against 类别对应位置为 1
            self.tensor_labels.append(one_hot)