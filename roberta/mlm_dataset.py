import torch
import torch.nn as nn
import pandas as pd
import random

class MLMDataset(torch.utils.data.Dataset):
    def __init__(self,df,tokenizer,max_seq_len=512, mask_prob=0.15):
        super().__init__()
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        self.csv_file = df
        self.length = self.get_file_length()
        self.mask_prob=mask_prob
        # self.data = pd.read_csv(self.csv_file,chunksize=1,iterator=True)
        
    def get_file_length(self):
        """Gets the number of rows in the CSV file."""
        chunk_iterator = pd.read_csv(self.csv_file, chunksize=1000) #read in chunks to get length.
        count = 0
        for chunk in chunk_iterator:
            count += len(chunk)
        return count
    
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        df_row = pd.read_csv(self.csv_file,chunksize=1,skiprows=idx,nrows=1)
        text = df_row.get_chunk().values[0][0]

        # text = self.data.get_chunk().values[0][0]
        tokenize = self.tokenizer(text,
                                truncation=True,
                                max_length=512,
                                padding='max_length',
                                padding_side='right',
                                return_tensors='pt')
        tokens = tokenize.input_ids.squeeze()
        # return token    

        attention_mask = tokenize.attention_mask.squeeze()

        labels = tokens.clone()
        # Masking
        mask = torch.rand(tokens.shape) < self.mask_prob
        masked_indices = torch.nonzero(mask).squeeze(-1)

        masked_tokens = tokens.clone()
        for index in masked_indices:
            rand = random.random()
            if rand < 0.8:  # 80% mask
                masked_tokens[index] = self.tokenizer.mask_token_id
            elif rand < 0.9:  # 10% random
                masked_tokens[index] = random.randint(0, self.tokenizer.vocab_size - 1)
            # 10% keep original

        return masked_tokens,attention_mask,labels
    

#For Hugginface Training
class MLMDatasetHF(torch.utils.data.Dataset):
    def __init__(self,df,tokenizer,max_seq_len):
        super().__init__()
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        self.csv_file = df
        self.length = self.get_file_length()
        self.data = pd.read_csv(self.csv_file,chunksize=1,iterator=True)

    def get_file_length(self):
        """Gets the number of rows in the CSV file."""
        chunk_iterator = pd.read_csv(self.csv_file, chunksize=1000) #read in chunks to get length.
        count = 0
        for chunk in chunk_iterator:
            count += len(chunk)
        return count
    
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        df_row = pd.read_csv(self.csv_file,chunksize=1,skiprows=idx,nrows=1)

        # text = df_row.get_chunk().values[0][0]
        text = self.data.get_chunk().values[0][0]
        token = self.tokenizer.encode(text,
                                      truncation=True,
                                      max_length=self.max_seq_len,
                                      padding='max_length',
                                      padding_side='right',
                                      return_tensors='pt').squeeze(0)
        return token    