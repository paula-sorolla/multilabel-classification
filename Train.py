from tqdm.notebook import tqdm
from helper import Classifier, Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from official.nlp import optimization  # to create AdamW optimizer
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

from src.Preprocessing import Preprocessing

class Train():
    def __init__(self, args, loss_fn) -> None:
        self.n_labels = 52
        self.config = {
            'learning_rate': 3e-5,
            'batch_size': 64,
            'epochs': 10,
            'dropout': 0.3,
            'tokenizer_max_len': 40,
            'truncate': True,
        }
        self.args = args
        self.loss_fn = loss_fn

    def trainer(self):
        train_dataset, valid_dataset = build_dataset(config['tokenizer_max_len'], self.config['truncate'])
        train_data_loader, valid_data_loader = build_dataloader(train_dataset, valid_dataset, config['batch_size'])
        print("Length of Train Dataloader: ", len(train_data_loader))
        print("Length of Valid Dataloader: ", len(valid_data_loader))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_train_steps = int(len(train_dataset) / config['batch_size'] * 10)

        model = ret_model(n_train_steps, self.config['dropout'])
        optimizer = self.ret_optimizer(model)
        scheduler = self.ret_scheduler(optimizer, n_train_steps)
        model.to(device)
        model = nn.DataParallel(model)
        
        n_epochs = self.config['epochs']
        loss_fn = self.config['loss']

        best_val_loss = 100
        for epoch in tqdm(range(n_epochs)):
            print('Train EPOCH: ', epoch)
            train_loss = self.train_fn(train_data_loader, model, loss_fn, optimizer, device, scheduler)
            eval_loss, preds, labels = self.eval_fn(valid_data_loader, model, device)
            
            metrics_eval = log_metrics(preds, labels)
            try:
                auc_score  = metrics_eval["auc_micro"]
    #             print("AUC score: ", auc_score)
            except:
                pass
            avg_train_loss, avg_val_loss = train_loss / len(train_data_loader), eval_loss / len(valid_data_loader)

            print("Average Train loss: ", avg_train_loss)
            print("Average Valid loss: ", avg_val_loss)
            torch.save(model.state_dict(), "./model_current.pt")  

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "./model_best.pt")  
                print("Model saved as current val_loss is: ", best_val_loss)    

    def train_fn(self, data_loader, model, optimizer, device, scheduler):
        """ 
        Train function
            Modified from Abhishek Thakur's BERT example: 
            https://github.com/abhishekkrthakur/bert-sentiment/blob/master/src/engine.py

        Args:
            data_loader (_type_): _description_
            model (_type_): _description_
            optimizer (_type_): _description_
            device (_type_): _description_
            scheduler (_type_): _description_

        Returns:
            _type_: _description_
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
        '''
            Modified from Abhishek Thakur's BERT example: 
            https://github.com/abhishekkrthakur/bert-sentiment/blob/master/src/engine.py
        '''
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
        model = Classifier(n_train_steps, self.n_labels, do_prob, bert_model=bert_model)
        return model

    def ret_optimizer(self, model):
        """ Taken from Abhishek Thakur's Tez library example: 
            https://github.com/abhishekkrthakur/tez/blob/main/examples/text_classification/binary.py

        Args:
            model (_type_): _description_

        Returns:
            _type_: _description_
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
        sch = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
        return sch
    
    def trainer(self):

        train_dataset, valid_dataset = build_dataset(self.config['tokenizer_max_len'])
        train_data_loader, valid_data_loader = build_dataloader(train_dataset, valid_dataset, self.config['batch_size'])
        print("Length of Train Dataloader: ", len(train_data_loader))
        print("Length of Valid Dataloader: ", len(valid_data_loader))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_train_steps = int(len(train_dataset) / self.config['batch_size'] * 10)

        model = ret_model(n_train_steps, self.config['dropout'])
        optimizer = ret_optimizer(model)
        scheduler = ret_scheduler(optimizer, n_train_steps)
        model.to(device)
        model = nn.DataParallel(model)
        
        n_epochs = self.config['epochs']

        best_val_loss = 100
        for epoch in tqdm(range(n_epochs)):
            print('Train EPOCH: ', epoch)
            train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
            eval_loss, preds, labels = eval_fn(valid_data_loader, model, device)
            
            auc_score = log_metrics(preds, labels)["auc_micro"]
            print("AUC score: ", auc_score)
            avg_train_loss, avg_val_loss = train_loss / len(train_data_loader), eval_loss / len(valid_data_loader)

            print("Average Train loss: ", avg_train_loss)
            print("Average Valid loss: ", avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "./best_model.pt")  
                print("Model saved as current val_loss is: ", best_val_loss)    