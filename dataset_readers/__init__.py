import os, sys

from dataset_readers.kbalbert_dataset_reader import KbAlbertDataset

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

__all__ = ['KbAlbertDataset']