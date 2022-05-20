import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from src.Preprocessing import Dataset

class Inference():
    def __init__(self, args, model, tokenizer, test_set) -> None:
        """Inference class for Test set analysis

        Args:
            args (ArgumentParser): Argument parser
            model (torch.nn): Neural Network model to predict
            tokenizer (AutoTokenizer): Hugging face tokenizer
            test_set (DataFrame): Test set dataframe
        """        
        self.args = args
        self.model = model
        self.test = test_set
        self.tokenizer = tokenizer

    def inference_batches(self, batchSize = 64):
        """Predict outputs for inference phase

        Args:
            batchSize (int, optional): Batch size. Defaults to 64.

        Returns:
            [torch]: Prediction outputs
            [torch]: Targets
        """        
        test_targets = []
        test_outputs = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the data
        test_dataset = Dataset(self.test.input.tolist(), self.test.iloc[:, 3:].values.tolist(), self.tokenizer, self.args['tokenizer_max_len'], self.args['truncate'])
        data_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True, num_workers=2)

        with torch.no_grad():
            for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
                ids = d["ids"]
                mask = d["mask"]
                labels = d["labels"]

                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)

                # Get NN outputs and store them
                outputs = self.model(ids=ids, mask=mask)
                test_targets.extend(labels.cpu().numpy())
                test_outputs.extend(torch.sigmoid(outputs).cpu().numpy())

        return test_outputs, test_targets