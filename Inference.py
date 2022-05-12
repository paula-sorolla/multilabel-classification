
from src.Preprocessing import Preprocessing
from src.Dataset import Dataset

from tqdm.notebook import tqdm

import torch
from torch.utils.data import DataLoader

class Inference():
    def __init__(self, args, model, tokenizer, test_set) -> None:
        self.args = args
        self.model = model
        self.test = test_set
        self.tokenizer = tokenizer

    # # Predict outputs:
    # test_clean = Preprocessing.remove_testDuplicates(train, val, test)
    # model = load_model()
    # preds, labels = inference_batches(test_clean, model)
    # all_metrics = get_metrics(preds, labels)

    def inference_batches(self, batchSize = 1024):
        '''
        Predict outputs for inference phase
        '''
        test_targets = []
        test_outputs = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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

                outputs = self.model(ids=ids, mask=mask)
                test_targets.extend(labels.cpu().numpy())
                test_outputs.extend(torch.sigmoid(outputs).cpu().numpy())


        return test_outputs, test_targets