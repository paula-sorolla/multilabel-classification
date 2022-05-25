from tqdm.notebook import tqdm
import os

import pandas as pd
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.Postprocessing import Metrics
from src.Preprocessing import Preprocessing
from src.BuildModels import create_model
from src.LossFunctions import LossFunctions, get_classWeigths

class Train(object):
    def __init__(self,args):
        self.n_labels = 52
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the LossFunctions object
        self.prep = Preprocessing(self.args.data_path)
        train_df, _, _ = self.prep.retrieve_sets()
        weights = get_classWeigths(train_df)
        self.LossF = LossFunctions(self.args.loss_function, weights.to(self.device, dtype=torch.float))
        
        # Call the trainer
        print('Starting the training!')
        train_loss, eval_loss = self.trainer()

        self.save_losses(train_loss, eval_loss)


    def trainer(self):
        """Train the model
        """   
        # Create the model
        model, tokenizer, optimizer, scheduler = create_model(self.args, 'allenai/scibert_scivocab_uncased')
        
        train_losses = {}
        val_losses = {}
        best_val_loss = 100
        for epoch in tqdm(range(self.args.num_epochs)):
            print('Train EPOCH: ', str(epoch+1))

            
            train_dataset, val_dataset, _ = self.prep.build_dataset(tokenizer, self.args.tokenizer_max)
            train_data_loader, valid_data_loader = self.prep.build_dataloader(train_dataset, val_dataset, self.args.batch_size)

            # Run train and validation
            train_loss = self.train_fn(train_data_loader, model, self.args.loss_function, optimizer, self.device, scheduler)
            eval_loss, preds, labels = self.eval_fn(valid_data_loader, model, self.args.loss_function, self.device)
            
            # Get metrics for validation set
            metr = Metrics(preds, labels, self.args.threshold)
            eval_metrics = metr.retrieve_allMetrics()
            print('Macro F1 score: ', eval_metrics['Macro F1 score'])

            # Get the average loss of the epoch
            avg_train_loss, avg_val_loss = train_loss / len(train_data_loader), eval_loss / len(valid_data_loader)
            print("Average Train loss: ", avg_train_loss)
            print("Average Valid loss: ", avg_val_loss)
            train_losses['Train'+str(epoch+1)] = avg_train_loss
            val_losses['Val'+str(epoch+1)] = avg_val_loss
            
            # Save the model at the current epoch
            p = os.path.join('models/', 'model_current.pt')
            torch.save(model.state_dict(), p)
            mlflow.log_artifact(p)

            # Save the model if the loss has decreased
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                p = os.path.join('models/', 'model_best.pt')
                torch.save(model.state_dict(), p)
                mlflow.log_artifact(p)
                print("Model saved as current val_loss is: ", best_val_loss)
        
        mlflow.log_metrics(train_losses)
        mlflow.log_metrics(val_losses)

        return train_losses, val_losses

    def train_fn(self, data_loader, model, loss_fn, optimizer, device, scheduler):
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

            loss = self.LossF.calculate_loss(outputs, targets)
            print('Training... ', bi, '. Loss... ', loss)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            scheduler.step()
        return train_loss
    

    def eval_fn(self, data_loader, model, loss_fn, device):
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
                loss = self.LossF.calculate_loss(outputs, targets)
                eval_loss += loss.item()
                fin_targets.extend(targets)
                fin_outputs.extend(torch.sigmoid(outputs))
        return eval_loss, fin_outputs, fin_targets

    def save_losses(train_loss, eval_loss):
        loss_epochs = pd.DataFrame([train_loss, eval_loss]).rename(index={0: "Train loss", 1: "Eval loss"}).T
        loss_epochs.to_csv('outputs/Losses.csv')
    