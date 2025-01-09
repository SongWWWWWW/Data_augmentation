import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description="Train a RoBERTa-based model with MLP classifier")
    
    # Add arguments for training configurations
    parser.add_argument('--num_labels', type=int, default=3, help="Number of labels for classification")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs for training")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate for optimizer")
    # parser.add_argument('--train_file', type=str, required=True, help="Path to the training dataset file")
    parser.add_argument('--gpu', action='store_true', help="Whether to use GPU or not")
    parser.add_argument('--train_path', type=str, help="the path of train set")
    parser.add_argument('--val_path', type=str, help="the path of val set")
    parser.add_argument('--save_path', type=str,help="the path of saving model")
    return parser.parse_args()

def eval_parse_args():
    parser = argparse.ArgumentParser(description="test a RoBERTa-based model with MLP classifier")
    
    parser.add_argument('--model_root_path', type=str, help="[./results/model/]checkpoint-xxx/modle.safetensors")
    parser.add_argument('--test_path', type=str, help="the path of test set")
    parser.add_argument('--output_error_question', type=lambda x: (str(x).lower() == 'true'), default=False,help="whether ouput mistake questions or not ")
    parser.add_argument('--save_log',type=lambda x: (str(x).lower() == 'true'), default=True,help= "save error file and f1 score")
    parser.add_argument('--sign',type=str, help="the sign of eval trial")
    return parser.parse_args()
