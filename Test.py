import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from src.Preprocessing import Preprocessing
from src.BuildModels import create_model


class Test(object):
    def __init__(self, args) -> None:
        """Inference class for Test set analysis

        Args:
            args (ArgumentParser): Argument parser
        """        
        self.args = args

        # Run the inference
        test_outputs, test_targets = self.inference_batches()
        return test_outputs, test_targets

    def inference_batches(self):
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
        
        # Create the model
        model, tokenizer, _, _ = create_model(self.args, 'allenai/scibert_scivocab_uncased').cuda()

        # Load the data
        prep = Preprocessing(self.args.data_path)
        _, _, test_dataset = prep.build_dataset(tokenizer, self.args.tokenizer_max)
        data_loader = prep.build_dataloader(test_dataset, None, self.args.batch_size)

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