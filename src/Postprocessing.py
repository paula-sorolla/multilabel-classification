import pandas as pd


class Metrics():    
    def __init__(self, preds, labels, threshold=0.5):
        """Class used to compute some metrics on multilabel results

        Args:
            preds (list): List of predicted outputs [1 x Num Classes]
            labels (list): List of ground true labels [1 x Num Classes]
        """        
        lab_df = pd.DataFrame(labels)
        # pred_df = pd.DataFrame(preds).round(0).astype(int)
        pred_df = pd.DataFrame(preds).applymap(lambda x: self.set_threshold(x, threshold))
        
        self.lab_df = lab_df
        self.pred_df = pred_df

        # Calculate tp/fp/fn/tn per class:
        self.tp = (pred_df + lab_df).eq(2).sum()
        self.fp = (pred_df - lab_df).eq(1).sum()
        self.fn = (pred_df - lab_df).eq(-1).sum()
        self.tn = (pred_df + lab_df).eq(0).sum()

    def get_classWeight(self):
        """Compute the class weight over all samples

        Returns:
            [list]: Proportion of occurrence of each of the classes over the whole samples
        """        
        return self.lab_df.sum() / sum(self.lab_df.sum())

    def set_threshold(x, thr):
        return 1 if x > thr else 0

    def get_precision(self, i):
        """Compute the precision of class i

        Args:
            i (int): Class index to be computed. Use -1 to compute micro scroe

        Returns:
            [float]: Precision value for the label
        """ 
        if i == -1:
            tp = sum(self.tp)
            fp = sum(self.fp)
        else:   
            tp = self.tp[i]
            fp = self.fp[i]
        
        return tp / (tp + fp) if tp + fp != 0 else 0.0

    def get_recall(self, i):
        """Compute recall of class i

        Args:
            i (int): Class index to be computed. Use -1 to compute micro scroe

        Returns:
            [float]: Recall value for class i
        """        
        if i == -1:
            tp = sum(self.tp)
            fn = sum(self.fn)
        else:   
            tp = self.tp[i]
            fn = self.fn[i]
        return tp / (tp + fn) if tp + fn != 0 else 0.0

    def get_f1(self, i):
        """Compute F1 score of class i

        Args:
            i (int): Class index to be computed. Use -1 to compute micro scroe

        Returns:
            _type_: _description_
        """     
        if i == -1:
            prec = self.get_precision(-1)
            rec = self.get_recall(-1)
            tp = sum(self.tp)
        else:   
            prec = self.get_precision(i)
            rec = self.get_recall(i)
            tp = self.tp[i]

        return 2 * prec * rec / (prec + rec) if tp > 0 else 0.0

    def get_macroAvg(metric):
        """Compute the macro average of a given metric (average over classes)

        Args:
            metric (list): List of scores for all classes

        Returns:
            [float]: Macro average
        """        
        macro_avg = sum(metric) / len(metric)
        return macro_avg

    def retrieve_allMetrics(self):
        """Retrieve all metrics
            - Metrics per class
            - Macro average over classes
            - Micro average over classes

        Returns:
            [dict]: Dictionary retrieving all metrics
        """        

        # Compute scores per class
        prec = [self.get_precision(i) for i in range(len(self.tp))]
        rec = [self.get_recall(i) for i in range(len(self.tp))]
        f1 = [self.get_f1(i) for i in range(len(self.tp))]

        # Macro scores:
        prec_mac = self.get_macroAvg(prec)
        rec_mac = self.get_macroAvg(rec)
        f1_mac = self.get_macroAvg(f1)

        # Micro scores:
        prec_mic = self.get_precision(-1)
        rec_mic = self.get_recall(-1)
        f1_mic = self.get_f1(-1)
        
        return {
            # Metrics per class:
            'Precision': prec, 'Recall': rec, 'F1 score': f1,
            # Macro scores:
            'Macro precision': prec_mac, 'Macro recall': rec_mac, 'Macro F1 score': f1_mac,
            # Micro scores:
            'Micro precision': prec_mic, 'Micro recall': rec_mic, 'Micro F1 score': f1_mic,
        }