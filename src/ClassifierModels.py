import torch.nn as nn

class Classifier(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """    
    def __init__(self, n_train_steps, n_classes, do_prob, bert_model):
        super(Classifier, self).__init__()
        self.bert = bert_model
        self.bert_hiddenSize = 768
        self.dropout = nn.Dropout(do_prob)
        self.out = nn.Linear(self.bert_hiddenSize, n_classes)
        self.n_train_steps = n_train_steps
        self.step_scheduler_after = "batch"

    def forward(self, ids, mask):
        output_1 = self.bert(ids, attention_mask=mask)["pooler_output"]
        output_2 = self.dropout(output_1)
        output = self.out(output_2)
        return output