import argparse
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
#mlflow
import mlflow
import mlflow.pytorch
mlflow.set_experiment("/Users/sorollabayodp/Documents/Thesis/7.Development/Experiments run")

# Own libraries
from src.Preprocessing import Preprocessing
from src.Postprocessing import Metrics
from src.utils import Map, get_argparse_defaults
from Test import Test
from Train import Train

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Multilabel classifier (BERT)')
    parser.add_argument('-tr', '--train',  default=True, help="If True then trains the model", type=bool)
    # parser.add_argument('-ts', '--test',  default=True, action=argparse.BooleanOptionalAction, help="If True then tests the model")
    parser.add_argument('-dp', '--data_path', default='/data/', type=str, help='Directory path to the dataset')
    parser.add_argument('-op', '--output_path', default='/outputs/', type=str, help='Directory path to the outputs (for saving/loading)')
    parser.add_argument('-lr', '--learning_rate', default=1e-5, type=float, help='Learning rate to use for model training')
    parser.add_argument('-mn', '--model_name', default='best_model.pt', type=str, help='Model name to be loaded')
    parser.add_argument('-j', '--workers', default=8, type=int, help='Number of data loading workers (default: 8)')
    parser.add_argument('-thr', '--threshold', default=0.5, type=float, help='Classificaiton threshold value')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('-tm', '--tokenizer_max', default=200, type=int, help='Tokenizer max length (default: 200)')
    parser.add_argument('-e', '--num_epochs', default=10, type=int, help='number of epochs (default: 10)')
    parser.add_argument('-lf', '--loss_function', default="BCE", type=str, help='loss function (default: BCE)')
    parser.add_argument('-do', '--do_prob', default=0.5, type=float, help='Dropout probability')
    # parser.add_argument('--slope', '-s', default=-1, type=float, help='Slope of the sigmoid function loss')
    # parser.add_argument('--offset', '-off', default=0, type=float, help='Offset of the sigmoid function loss')

    args = parser.parse_args()

    # Log the experiment parameters
    for arg in vars(args):
        mlflow.log_param(arg, getattr(args, arg))

    # Do train, test, or both 
    if args.train:
        class_train = Train(args)
    if args.test:
        labels, outputs = Test(args)
        test_metrics = Metrics(outputs, labels, args.threshold).retrieve_allMetrics()

    print('Done!')

