import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import re
import pandas as pd

import torch
from torch.utils.data import DataLoader

class Dataset:
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels

        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        inputs = self.tokenizer.__call__(text,
                                        None,
                                        add_special_tokens=True,
                                        max_length=self.max_len,
                                        padding="max_length",
                                        truncation=True,
                                        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class Preprocessing(object):

    def __init__(self, path):
        self.path = path
        self.n_labels = 52

        self.train, self.test, self.val = self.load_dataframes()

    def retrieve_sets(self):
        return self.train, self.test, self.val

    def load_dataframes(self):
        """ Load train, test and validation dataframes to run the experiments

        Returns:
            [DataFrame]: Train DataFrame
            [DataFrame]: Test DataFrame
            [DataFrame]: Validation DataFrame
        """        

        # Load data from Set A, B and EX
        train_A, test_A, val_A = self.load_data(self.path, version="A", suffix="_kw", reduce_memory=True)
        train_B, test_B, val_B = self.load_data(self.path, version="B", suffix="_kw", reduce_memory=True)
        train_EX, test_EX, val_EX = self.load_data(self.path, version="EX", suffix="_kw", reduce_memory=True)

        # Build the train, test and val datasets:
        train = pd.concat([train_A, train_B, train_EX])
        test = pd.concat([test_A, test_B, test_EX])
        val = pd.concat([val_A, val_B, val_EX])

        return train, test, val

    
    def reduce_mem_usage(self, df, verbose=True):
        """ Helper function to reduce the memory of Pandas DataFrame by converting the column types.

        Args:
            df (DataFrame): Pandas DataFrame to convert
            verbose (bool, optional): Defaults to True.

        Returns:
            DataFrame: converted Pandas DataFrame 
        """      

        numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2    
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df

    def load_data(self, path="data/", version="A", suffix="_kw", reduce_memory=True):
        """ Helper function to load training, test and validation set.

        Args:
            path (str, optional): Data location path. Defaults to "data/".
            version (str, optional): Set to load. Defaults to "A".
            suffix (str, optional): suffix. Defaults to "_kw".
            reduce_memory (bool, optional): Reduce memory of the Pandas DataFrame by  converting the column types. Defaults to True.

        Returns:
            train: Train DataFrame for the specified set
            test: Test DataFrame for the specified set
            val: Validation DataFrame for the specified set
        """        
        
        train = pd.read_csv(f"{path}set_{version}_train{suffix}.csv")
        test = pd.read_csv(f"{path}set_{version}_test{suffix}.csv")
        val = pd.read_csv(f"{path}set_{version}_val{suffix}.csv")
        
        # Reduce memory by optimizing DF columns types if wanted
        if reduce_memory:
            train = self.reduce_mem_usage(train)
            test = self.reduce_mem_usage(test)
            val = self.reduce_mem_usage(val)
        
        print(f"Set {version} with suffix '{suffix}' was loaded successfully.")
        return train, test, val

    def build_dataset(self, tokenizer, tokenizer_max_len, truncate):
        
        '''
        Tokenize and map the training and validation sets
        '''
        train_dataset = Dataset(self.train.input.tolist(), self.train.iloc[:, 3:].values.tolist(), tokenizer, tokenizer_max_len, truncate)
        valid_dataset = Dataset(self.val.input.tolist(), self.val.iloc[:, 3:].values.tolist(), tokenizer, tokenizer_max_len, truncate)
        
        return train_dataset, valid_dataset

    def build_dataloader(train_dataset, valid_dataset, batch_size):
        '''
        Create the torch dataloaders
        '''
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        return train_data_loader, valid_data_loader

    @staticmethod
    def remove_testDuplicates(train, val, test):
        '''
        Remove duplicates of train/val datasets present in the test set
        '''
        # Get the training duplicates:
        duplicates_train = set(test.pui) & set(train.pui) 
        test_clean = test[~test['pui'].isin(duplicates_train)]
        
        # Get the validation duplicates:
        duplicates_val = set(test.pui) & set(val.pui) 
        test_clean = test_clean[~test_clean['pui'].isin(duplicates_val)]
        
        try:
            assert test_clean.shape[0] == test.shape[0] - len(duplicates_train) - len(duplicates_val)
            return test_clean
        
        except:
            return test

class Dataset:
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels

        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        inputs = self.tokenizer.__call__(text,
                                        None,
                                        add_special_tokens=True,
                                        max_length=self.max_len,
                                        padding="max_length",
                                        truncation=True,
                                        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }