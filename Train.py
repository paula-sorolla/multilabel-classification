from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

from src.Preprocessing import Preprocessing
from src.Postprocessing import Metrics
from src.ClassifierModels import Classifier, LSTM_Classifier

class Train():
    def __init__(self, args, loss_fn) -> None:
        self.n_labels = 52
        self.args = args
        self.loss_fn = loss_fn

    def trainer(self):
        """Train the model
        """        
        train_dataset, valid_dataset = Preprocessing.build_dataset(self.args.tokenizer_max_len, self.args.truncate)
        train_data_loader, valid_data_loader = Preprocessing.build_dataloader(train_dataset, valid_dataset, self.args.batch_size)
        print("Length of Train Dataloader: ", len(train_data_loader))
        print("Length of Valid Dataloader: ", len(valid_data_loader))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_train_steps = int(len(train_dataset) / self.args.batch_size * 10)

        model = Preprocessing.ret_model(n_train_steps, self.args.dropout)
        optimizer = self.ret_optimizer(model)
        scheduler = self.ret_scheduler(optimizer, n_train_steps)
        model.to(device)
        model = nn.DataParallel(model)

        best_val_loss = 100
        for epoch in tqdm(range(self.args.epochs)):
            print('Train EPOCH: ', epoch)
            train_loss = self.train_fn(train_data_loader, model, self.args.loss, optimizer, device, scheduler)
            eval_loss, preds, labels = self.eval_fn(valid_data_loader, model, device)
            
            eval_metrics = Metrics(preds, labels).retrieve_allMetrics()
            print('Macro F1 score: ', eval_metrics['Macro F1 score'])

            avg_train_loss, avg_val_loss = train_loss / len(train_data_loader), eval_loss / len(valid_data_loader)

            print("Average Train loss: ", avg_train_loss)
            print("Average Valid loss: ", avg_val_loss)
            torch.save(model.state_dict(), "./models/model_current.pt")  

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "./models/model_best.pt")  
                print("Model saved as current val_loss is: ", best_val_loss)    

    def train_fn(self, data_loader, model, optimizer, device, scheduler):
        """ 
        Train function
            Modified from Abhishek Thakur's BERT example: 
            https://github.com/abhishekkrthakur/bert-sentiment/blob/master/src/engine.py

        Args:
            data_loader (DataLoader): PyTorch DataLoader
            model (torch.nn): PyTorch NN model to evaluate
            device (torch.device): Selected device for PyTorch operations
            optimizer: AdamW optimizer
            scheduler: Transformer scheduler

        Returns:
            [Tensor]: Train loss
        """    

        train_loss = 0.0
        model.train()
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["labels"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(ids=ids, mask=mask)

            loss = self.loss_fn(outputs, targets)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            scheduler.step()
        return train_loss
    

    def eval_fn(self, data_loader, model, device):
        """ Modified from Abhishek Thakur's BERT example: 
            https://github.com/abhishekkrthakur/bert-sentiment/blob/master/src/engine.py

        Args:
            data_loader (DataLoader): PyTorch DataLoader
            model (torch.nn): PyTorch NN model to evaluate
            device (torch.device): Selected device for PyTorch operations

        Returns:
            [Tensor]: Evaluation loss
        """        
        eval_loss = 0.0
        model.eval()
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
                ids = d["ids"]
                mask = d["mask"]
                targets = d["labels"]

                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.float)

                outputs = model(ids=ids, mask=mask)
                loss = self.loss_fn(outputs, targets)
                eval_loss += loss.item()
                fin_targets.extend(targets)
                fin_outputs.extend(torch.sigmoid(outputs))
        return eval_loss, fin_outputs, fin_targets
    
    def ret_model(self, n_train_steps, do_prob):
        """Retrieve NN module

        Args:
            n_train_steps (int):  Number of training steps
            do_prob (float): Dropout probability (tunable parameter)

        Returns:
            model: NN model to be trained
        """        
        bert_model = transformers.AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        model = Classifier(n_train_steps, self.n_labels, do_prob, bert_model=bert_model)
        return model

    def ret_optimizer(self, model):
        """ Retrieve optimizer
            Library example: https://github.com/abhishekkrthakur/tez/blob/main/examples/text_classification/binary.py

        Args:
            model (torch.nn): NN model to optimize

        Returns:
            AdamW: AdamW optimizer
        """        

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=self.config['learning_rate'])
        return opt

    def ret_scheduler(optimizer, num_train_steps):
        """ Retrieve a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
             after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
             [https://huggingface.co/docs/transformers/main_classes/optimizer_schedules]

        Args:
            optimizer: NN trainer optimizer
            num_train_steps (int): Number of training steps

        Returns:
            Transformer scheduler
        """        
        sch = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
        return sch