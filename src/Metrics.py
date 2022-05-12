import pandas as pd


class Metrics():
    def __init__(self, preds, labels):
        # super().__init__()
        lab_df = pd.DataFrame(labels)
        pred_df = pd.DataFrame(preds).round(0).astype(int)
        self.lab_df = lab_df
        self.pred_df = pred_df

        # Calculate tp/fp/fn/tn per class:
        self.tp = (pred_df + lab_df).eq(2).sum()
        self.fp = (pred_df - lab_df).eq(1).sum()
        self.fn = (pred_df - lab_df).eq(-1).sum()
        self.tn = (pred_df + lab_df).eq(0).sum()

    def get_classWeight(self):
        return self.lab_df.sum() / sum(self.lab_df.sum())

    def get_precision(self):
        prec = [self.tp[i] / (self.tp[i] + self.fp[i]) * 100.0 if self.tp[i] + self.fp[i] != 0 else 0.0 for i in range(len(self.tp))]
        return prec

    def get_recall(self):
        rec = [self.tp[i] / (self.tp[i] + self.fn[i]) * 100.0 if self.tp[i] + self.fn[i] != 0 else 0.0 for i in range(len(self.tp))]
        return rec

    def get_f1(self, weighted=False):
        prec = self.get_precision()
        rec = self.get_recall()
        if weighted:
            weight = self.get_classWeight()
            f1 = [weight[i] * 2 * prec[i] * rec[i] / (prec[i] + rec[i]) if self.tp[i] > 0 else 0.0 for i in range(len(self.tp))]
        else:
            f1 = [2 * prec[i] * rec[i] / (prec[i] + rec[i]) if self.tp[i] > 0 else 0.0 for i in range(len(self.tp))]
        
        return f1

    def get_macroAvg(metric):
        macro_avg = sum(metric) / len(metric)
        return macro_avg

    def retrieve_allMetrics(self):
        '''
        Create some metrics: precison, recall, F1...
        '''
        
        # Calculate precision and recall:
        precision = self.get_precision()
        recall = self.get_recall()
        
        # Calculate F1 score:
        f1_score = self.get_f1(weighted=False)
        f1_scoreWeighted = self.get_f1(weighted=True)
        
        # Macro average:
        prec_avg = self.get_macroAvg(precision)
        rec_avg = self.get_macroAvg(recall)
        f1_avg = self.get_macroAvg(f1_score)
        f1wgt_avg = self.get_macroAvg(f1_scoreWeighted)
        
        return {
            'Precision': precision,
            'Recall': recall,
            'F1 score': f1_score,
            'Weighted F1 score': f1_scoreWeighted,
            'Average precision': prec_avg,
            'Average recall': rec_avg,
            'Average F1 score': f1_avg,
            'Average weighted F1 score': f1wgt_avg,
        }