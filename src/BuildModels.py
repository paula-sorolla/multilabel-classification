import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

from src.Preprocessing import Preprocessing


def create_model(args, bert_hf):
    """Retrieve the NN model instance to be trained/tested

    Args:
        args (): _description_
        bert_hf (_type_): _description_

    Returns:
        model: NN model to be trained
    """    
    # Load the BERT Model and Tokenizer to be used:
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_hf, do_lower_case=True)
    bert_model = transformers.AutoModel.from_pretrained(bert_hf)
    for param in bert_model.parameters():
        param.requires_grad = False

    prep = Preprocessing(args.data_path)
    train_dataset, _, _ = prep.build_dataset(tokenizer, args.tokenizer_max)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_train_steps = int(len(train_dataset) / args.batch_size * 10)

    model = Classifier(args, bert_model, n_train_steps)
    optimizer = ret_optimizer(model, args.learning_rate)
    scheduler = ret_scheduler(optimizer, n_train_steps)
    model.to(device)
    model = nn.DataParallel(model)
    
    print("Model created!")
    return model, tokenizer, optimizer, scheduler

def ret_optimizer(model, lr):
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
    opt = AdamW(optimizer_parameters, lr=lr)
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

class Classifier(nn.Module):
    """ Simple Fintunable BERT classifier
    """    
    def __init__(self, args, bert_model, n_train_steps):
        """Initializes the neural network Classifier

        Args:
            n_train_steps (int): Number of training steps (train dataset length / batch size)
            n_classes (int): Number of classes to predict
            do_prob (float): Dropout probability (finetunable parameter)
            bert_model (transformers.AutoModel): BERT HuggingFace model
        """        
        super(Classifier, self).__init__()

        self.n_classes = 52
        self.bert_hiddenSize = 768
        self.bert = bert_model

        self.dropout = nn.Dropout(args.do_prob)
        self.out = nn.Linear(self.bert_hiddenSize, self.n_classes)
        self.n_train_steps = n_train_steps
        self.step_scheduler_after = "batch"

    def forward(self, ids, mask):
        """NN Forward pass: we receive the Tensor inputs and return a Tensor containing the output.

        Args:
            ids (tensor): Input IDS
            mask (tensor): Input Attention mask

        Returns:
            [tensor]: NN output
        """        
        output_1 = self.bert(ids, attention_mask=mask)["pooler_output"]
        output_2 = self.dropout(output_1)
        output = self.out(output_2)
        return output

class LSTM_Classifier(nn.Module):
    """BERT classifier followed by an LSTM layer and a classificaton head
    """    
    def __init__(self, n_train_steps, n_classes, do_prob, bert_model, max_tok, dimension=128):
        """Initializes the neural network BERT-LSTM Classifier

        Args:
            n_train_steps (int): Number of training steps (train dataset length / batch size)
            n_classes (int): Number of classes to predict
            do_prob (float): Dropout probability (finetunable parameter)
            bert_model (transformers.AutoModel): BERT HuggingFace model
            max_tok (int): Max tokenizer length
            dimension (int, optional): Output dimension of the LSTM layer. Defaults to 128.
        """        
        super(LSTM_Classifier, self).__init__()
        self.bert = bert_model
        self.dimension = dimension
        self.n_train_steps = n_train_steps
        self.n_classes = n_classes
        self.lstm = nn.LSTM(input_size=max_tok,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=do_prob)

        self.fc = nn.Linear(2*dimension, n_classes)
        
    def forward(self, ids, mask, text_len=200):    
        """NN Forward pass: we receive the Tensor inputs and return a Tensor containing the output.

        Args:
            ids (tensor): Input IDS
            mask (tensor): Input Attention mask
            text_len (int, optional): Input text length (used to pack the BERT output sequence). Defaults to 200.

        Returns:
            [tensor]: NN output
        """ 
        text_emb = self.bert(ids, attention_mask=mask)["pooler_output"]
        
        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_input = text_emb
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:] # Take only CLS token
        out_reduced = torch.cat((out_forward, out_reverse), 1) # Concatenates the given sequence of seq tensors in dim 1.
        text_fea = self.drop(out_reduced) # Dropout

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, self.n_classes)
        text_out = torch.sigmoid(text_fea)

        return text_out
        