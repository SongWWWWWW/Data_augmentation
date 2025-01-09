import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class RoBERTa_MLP(nn.Module):
    def __init__(self, num_labels):
        super(RoBERTa_MLP, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.fc1 = nn.Linear(self.roberta.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, num_labels)
        self.dropout = nn.Dropout(0.3)
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, input_ids, attention_mask, labels=None):

        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Take the [CLS] token output (the first token)
        cls_output = roberta_output.last_hidden_state[:, 0, :]
        # MLP layers
        x = self.dropout(cls_output)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        if labels is not None:
            loss = self.loss_fn(x, labels)
            # print(loss)
            return {"loss": loss, "logits": x}
        
        return {"logits": x}

    
if __name__ == "__main__":
    model = RoBERTa_MLP(num_labels=1)
