from torch.nn import nn
import torch


class LossFunctions():
    def __init__(self, outputs, labels):
        super(LossFunctions, self).__init__()
        self.outputs = outputs
        self.labels = labels

    def BinaryCrossEntropy(self):
        """ Binary Cross Entropy loss

        Args:
            outputs (_type_): Output of the NN
            labels (_type_): Target labels, same shape as the outputs

        Returns:
            [float]: Loss
        """

        if self.labels is None:
            return None
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(self.outputs, self.labels.float()) 

    
    # class sigmoidF1(nn.Module): !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # @torch.cuda.amp.autocast() !!!!!! WHAT IS THIS??
        # def forward():

    def SigmoidF1(self, S=-1, E=0):
        y_hat = torch.sigmoid(self.outputs)
        y = self.labels

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

    def FocalLoss(self, gamma, alpha):
        y_hat = torch.sigmoid(self.outputs)
        y = self.labels

        if gamma and gamma < 0:
            raise ValueError("Value of gamma should be greater than or equal to zero.")        

        ce = self.BinaryCrossEntropy(y, y_hat)

        p_t = (y * y_hat) + ((1 - y) * (1 - y_hat))

        alpha_factor = y * alpha + (1 - y) * (1 - alpha)
        modulating_factor = torch.pow((1.0 - p_t), gamma)

        focal_loss = torch.sum(alpha_factor * modulating_factor * ce)

        return focal_loss
    