import json
import logging
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import AlbertTokenizer

logger = logging.getLogger(__name__)


class KbAlbertDataset(Dataset):
    def __init__(self,
                 file_path: str = None,
                 tokenizer: AlbertTokenizer = None,
                 max_length: int = 512) -> None:

        logger.info(f'Reading file at {file_path}')

        with open(file_path) as dataset_file:
            self.dataset = dataset_file.readlines()

        logger.info('Reading the dataset')

        self.processed_dataset = []

        for line in tqdm(self.dataset, desc='Processing'):
            data = json.loads(line)
            processed_data = {}
            encoded_dict = tokenizer(data['text'],
                                     add_special_tokens=True,
                                     max_length=max_length,
                                     truncation=True,
                                     padding='max_length',
                                     return_attention_mask=True,
                                     return_tensors='pt'
                                     )
            processed_data['input_ids'] = encoded_dict['input_ids'].squeeze(0)
            processed_data['attention_mask'] = encoded_dict['attention_mask'].squeeze(0)

            processed_data['label_major'] = torch.LongTensor([int(data['label_major'])])
            processed_data['label_minor'] = torch.LongTensor([int(data['label_minor'])])

            self.processed_dataset.append(processed_data)

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self,
                    idx: int = None):
        return self.processed_dataset[idx]