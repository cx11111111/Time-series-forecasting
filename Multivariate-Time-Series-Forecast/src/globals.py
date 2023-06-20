import torch
from torch.nn import MSELoss
from argparse import ArgumentParser

# Device on which we port the model and variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loss function: L1loss (mean absolute error)
loss_func = MSELoss()

parser = ArgumentParser()
parser.add_argument("--mode", type=str, choices=["train", "eval"],
                    default="train", help="Are we training or testing?")
parser.add_argument("--pretrained_path", type=str,
                    default="pretrained/pollution_net.pt",
                    help="Location of the pretrained net")
# DATASET PREPARATION ARGUMENTS
parser.add_argument("--dataset_path", type=str, default="C:\\Users\\cx\\Desktop\\数据集\\wtbdata_245days.csv",
                    help="Location of the dataset file")
parser.add_argument("--lag", type=int, default=24,
                    help="Time lag used for preparing train and test X-Y pairs")
parser.add_argument("--train_ratio", type=float, default=0.6,
                    help="Ratio for extracting the train set")
parser.add_argument("--val_ratio", type=float, default=0.2,
                    help="Ratio for extracting the validation set")
# MODEL ARGUMENTS
parser.add_argument("--hidden_dim", type=int, default=64,
                    help="Dimension of the LSTM hidden state vector")
#模型输出的维度
parser.add_argument("--output_size",type=int,default=1)
parser.add_argument("--num_layers", type=int, default=1,
                    help="Number of LSTM layers")
#多头注意力头的个数
parser.add_argument("--num_heads",type=int,default=4)
# TRAINING ARGUMENTS
parser.add_argument("--epochs", type=int, default=50,
                    help="Number of training epochs")
parser.add_argument("--bs", type=int, default=64,
                    help="Batch size")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning rate")
parser.add_argument("--wd", type=float, default=3e-10,
                    help="L2 regularization weight decay")
parser.add_argument("--dropout",type=float,default=0.001)
# Used to store training/model/dataset hyperparameters
config = parser.parse_args()
