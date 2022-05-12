import argparse
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

#mlflow
import mlflow
import mlflow.pytorch
mlflow.set_experiment("/Users/sorollabayodp/Documents/Thesis/7.Development/Experiments run")

from src.Preprocessing import Preprocessing
from Inference import Inference
from Train import Train


parser = argparse.ArgumentParser(description='PyTorch Multilabel classifier (BERT)')
parser.add_argument('--data', help='path to dataset', default='/data', type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--model-name', default='best_model.pt')
parser.add_argument('--model-path', default='/models/', type=str)
parser.add_argument('--num-classes', default=52)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--thre', default=0.5, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--num-epochs', '-e', default=1, type=int,
                    metavar='N', help='number of epochs (default: 1)')
parser.add_argument('--stop-epoch', '-se', default=40, type=int,
                    metavar='N', help='? stop epoch ? (default: 40)')
parser.add_argument('--weight-decay', '-wd', default=1e-4, type=int,
                    metavar='N', help='weight decay (default: 1e-4)')
parser.add_argument('--loss-function', '-lo', default="ASL", type=str,
                    metavar='N', help='loss function (default: ASL)')
parser.add_argument('--slope', '-s', default=-1, type=float,
                    metavar='N', help='Slope of the sigmoid function loss')
parser.add_argument('--offset', '-off', default=0, type=float,
                    metavar='N', help='Offset of the sigmoid function loss')


def main( data = '/data/',  ep = 1, loss = "BCE", num_classes = 52, E = 1, S = -9, batch_size = 64):

    args = get_argparse_defaults(parser)
    args = Map(args)

    args.num_epochs = ep
    args.loss_function = loss
    args.data = data
    args.num_classes = num_classes

    args.E = E
    args.S = S

    # args.model_path = args.model_path + model_file_name + ".pth"
    args.batch_size = batch_size
    print(args)

    for key, value in args.items(): #vars(args).items()
        mlflow.log_param(key, value)

    tokenizer = transformers.SqueezeBertTokenizer.from_pretrained("squeezebert/squeezebert-uncased", do_lower_case=True)

    # Setup model:
    print('creating model...')
    preprocess = Preprocessing(path=args.data)
    train, val, test =  preprocess.retrieve_sets()
    model = create_model(args).cuda()

    # Train:
    gg = Train(args).trainer()

    # Inference:
    labels, outputs = Inference(args, model, tokenizer, test).inference_batches()

