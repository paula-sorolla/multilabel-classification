from multiprocessing.sharedctypes import Value
import torch.nn as nn
import torch


class LossFunctions():
    def __init__(self, loss_fn, weights=None):
        """Loss calculations

        Args:
            outputs (_type_): _description_
            labels (_type_): _description_
        """        
        super(LossFunctions, self).__init__()

        self.loss_fn = loss_fn
        self.weights = weights
        if loss_fn == 'WeightedSigF1'and not weights:
            raise ValueError('Class weights could not be defined')

    def calculate_loss(self, outputs, labels):
        """Returns the loss according to the selected loss function

        Raises:
            ValueError: The loss function needs to match one of the available definitions

        Returns:
            [float]: Loss value
        """        

        match self.loss_fn:
            case 'BCE':
                return self.BinaryCrossEntropy(outputs, labels)
            case 'SigF1':
                return self.SigmoidF1(outputs, labels)
            case 'WeightedSigF1':
                return self.WeightedSigmoidF1(outputs, labels)
            case 'FL':
                return self.FocalLoss(outputs, labels)
            case _:
                raise ValueError("Please provide a valid loss function.")        

    def BinaryCrossEntropy(self, outputs, labels):
        """ Binary Cross Entropy loss (BCE)

        Returns:
            [float]: Computed BCE loss value
        """

        if labels is None:
            return None
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(outputs, labels.float()) 

    
    # @torch.cuda.amp.autocast()
    def SigmoidF1(self, outputs, labels, S=-1, E=0):
        """Sigmoid F1 loss function (https://arxiv.org/pdf/2108.10566.pdf)

        Args:
            S (float, optional): Sigmoid tunable parameter for the slope . Defaults to -1.
            E (float, optional): Sigmoid tunable parameter for the offset. Defaults to 0.

        Returns:
            [float]: Computed SigmoidF1 loss value
        """           
        y_hat = torch.sigmoid(outputs)
        y = labels

        # Sigmoid hyperparams:
        b = torch.tensor(S)
        c = torch.tensor(E)

        # Calculate the sigmoid
        sig = 1 / (1 + torch.exp(b * (y_hat + c)))

        tp = torch.sum(sig * y, dim=0)
        fp = torch.sum(sig * (1 - y), dim=0)
        fn = torch.sum((1 - sig) * y, dim=0)

        sigmoid_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        cost = 1 - sigmoid_f1
        macroCost = torch.mean(cost)

        return macroCost

    def Weighted_SigmoidF1(self, outputs, labels, S=-1, E=0):
        """Weighted Sigmoid F1 loss function (modification of https://arxiv.org/pdf/2108.10566.pdf)

        Args:
            S (float, optional): Sigmoid tunable parameter for the slope . Defaults to -1.
            E (float, optional): Sigmoid tunable parameter for the offset. Defaults to 0.

        Returns:
            [float]: Computed Weighted SigmoidF1 loss value to account for class imbalance
        """           
        y_hat = torch.sigmoid(outputs)
        y = labels

        # Sigmoid hyperparams:
        b = torch.tensor(S)
        c = torch.tensor(E)

        # Calculate the sigmoid
        sig = 1 / (1 + torch.exp(b * (y_hat + c)))

        tp = torch.sum(sig * y, dim=0)
        fp = torch.sum(sig * (1 - y), dim=0)
        fn = torch.sum((1 - sig) * y, dim=0)

        sigmoid_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        cost = 1 - sigmoid_f1
        weighted_cost = torch.mul(cost, self.weights)
        macroCost = torch.mean(weighted_cost)

        return macroCost


    def FocalLoss(self, outputs, labels, gamma=2.0, alpha=0.5):
        """ Focal loss function (https://arxiv.org/pdf/1708.02002.pdf)

        Args:
            gamma (float, optional): Focusing parameter. Defaults to 2.
            alpha (float, optional): Weighting factor. Defaults to 0.5.

        Raises:
            ValueError: Minimum value for the focusing parameter is 0

        Returns:
            [float]: Computed focal loss value
        """   
        y_hat = torch.sigmoid(outputs)
        y = labels

        if gamma and gamma < 0:
            raise ValueError("Value of gamma should be greater than or equal to zero.")        

        ce = self.BinaryCrossEntropy(y, y_hat)

        p_t = (y * y_hat) + ((1 - y) * (1 - y_hat))

        alpha_factor = y * alpha + (1 - y) * (1 - alpha)
        modulating_factor = torch.pow((1.0 - p_t), gamma)

        focal_loss = torch.sum(alpha_factor * modulating_factor * ce)

        return focal_loss
    
def get_classWeigths(train_df):
    
    train_labels = train_df.iloc[:, 3:]
    tot = sum(train_labels.sum(axis=0))
    weight = 1 - (train_labels.sum(axis=0) / tot)
    
    return torch.tensor(weight)