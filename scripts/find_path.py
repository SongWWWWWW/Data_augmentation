# 找到最好的那个model path, 并且记录最好的dev checkpoint and test f1 score
import json
import argparse 
import os
def parse_args():
    parser = argparse.ArgumentParser(description="test a RoBERTa-based model with MLP classifier")
    parser.add_argument('--model_root_path', type=str, help="[./results/model/]checkpoint-xxx/modle.safetensors")
    return parser.parse_args()
def find_path(model_root_path):
    with open(os.path.join(model_root_path,"logs","train_process_log.json"),"r") as f:
        dev = json.load(f)

    max_f1_score = max(dev["eval_f1_scores"])

    best_checkpoint = ""
    with open(os.path.join(model_root_path,"logs","log_dev_eval_log.json"),"r" ) as f:
        checkpoints = json.load(f)
        
    for k,v in checkpoints.items():
        if isinstance(v,dict):
            if v["F1 score"] == max_f1_score:
                best_checkpoint = k
            
    test_f1_score = 0

    with open(os.path.join(model_root_path,"logs","log_test_eval_log.json"), "r") as f:
        test = json.load(f)
        
    test_f1_score = test[best_checkpoint]["F1 score"]
    dic = {
        "best epoch": best_checkpoint,
        "dev_f1": max_f1_score,
        "test_f1": test_f1_score
    }
    # print(dic)
    with open(os.path.join(model_root_path,"logs","best_epoch.json"),"w") as f:
        json.dump(dic,f,indent=4)
    return best_checkpoint 

if __name__ == "__main__":
    args = parse_args() 
    with open(os.path.join(args.model_root_path,"logs","train_process_log.json"),"r") as f:
        dev = json.load(f)

    max_f1_score = max(dev["eval_f1_scores"])

    base_checkpoint = ""
    with open(os.path.join(args.model_root_path,"logs","log_dev_eval_log.json"),"r" ) as f:
        checkpoints = json.load(f)
        
    for k,v in checkpoints.items():
        if isinstance(v,dict):
            if v["F1 score"] == max_f1_score:
                best_checkpoint = k
            
    test_f1_score = 0

    with open(os.path.join(args.model_root_path,"logs","log_test_eval_log.json"), "r") as f:
        test = json.load(f)
        
    test_f1_score = test[best_checkpoint]["F1 score"]
    dic = {
        "best epoch": best_checkpoint,
        "dev_f1": max_f1_score,
        "test_f1": test_f1_score
    }
    # print(dic)
    with open(os.path.join(args.model_root_path,"logs","best_epoch.json"),"w") as f:
        json.dump(dic,f,indent=4)
    