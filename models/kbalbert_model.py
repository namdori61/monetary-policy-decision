from typing import Any, Tuple, Dict, Union, List, Optional, Sequence
import json

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy, precision, recall
from pytorch_lightning import TrainResult, EvalResult
from transformers import AlbertTokenizer, AlbertConfig, AlbertModel, AdamW

from dataset_readers import KbAlbertDataset


class KbAlbertClassificationModel(LightningModule):
    def __init__(self,
                 train_path: str = None,
                 dev_path: str = None,
                 test_path: str = None,
                 model_path: str = None,
                 config_path: str = None,
                 tokenizer: AlbertTokenizer = None,
                 num_classes: int = 2,
                 cuda_device: int = 0,
                 batch_size: int = 4,
                 num_workers: int = 0,
                 lr: float = 2e-5,
                 weight_decay: float = 0.1,
                 warm_up: int = 20):
        super(KbAlbertClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.cuda_device = cuda_device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.weight_decay = weight_decay
        self.warm_up = warm_up

        self.save_hyperparameters()

        self.train_dataset = KbAlbertDataset(train_path, tokenizer)
        self.val_dataset = KbAlbertDataset(dev_path, tokenizer)
        self.test_dataset = KbAlbertDataset(test_path, tokenizer)

        f = open(config_path, encoding='UTF-8')
        config_dict = json.loads(f.read())
        config = AlbertConfig(**config_dict)
        self.text_embedding = AlbertModel.from_pretrained(pretrained_model_name_or_path=model_path,
                                                          config=config)

        self.classifier_hidden_size = self.text_embedding.config.hidden_size
        self.classifier = nn.Linear(self.classifier_hidden_size, self.num_classes)

    def forward(self,
                batch: Dict = None) -> float:
        text_embedded = self.text_embedding(batch['input_ids'],
                                            token_type_ids=None,
                                            attention_mask=batch['attention_mask'])

        logits = self.classifier(text_embedded[1])

        return logits

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.cuda_device > 0:
            sampler = DistributedSampler(self.train_dataset)
        else:
            sampler = RandomSampler(self.train_dataset)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=sampler,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers)
        return train_dataloader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        sampler = SequentialSampler(self.val_dataset)

        val_dataloader = DataLoader(self.val_dataset,
                                    sampler=sampler,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers)
        return val_dataloader

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        sampler = SequentialSampler(self.test_dataset)

        test_dataloader = DataLoader(self.test_dataset,
                                    sampler=sampler,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers)
        return test_dataloader

    def configure_optimizers(self) -> Optional[
        Union[
            Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]
        ]
    ]:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.lr,
                          eps=1e-8)
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < self.warm_up:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.warm_up))
        else:
            lr_scale = min(1., float(self.warm_up) / float(self.trainer.global_step + 1))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_scale * self.lr

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def training_step(self,
                      batch: Dict = None,
                      batch_idx: int = None) -> Union[
        int, Dict[
            str, Union[
                Tensor, Dict[str, Tensor]
            ]
        ]
    ]:
        logits = self.forward(batch)
        if self.num_classes == 3:
            labels = batch['label_major']
        else:
            labels = batch['label_minor']
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        return {'loss': loss}

    def training_epoch_end(
            self, outputs: Union[TrainResult, List[TrainResult]]
    ) -> Dict[str, Dict[str, Tensor]]:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        logs = {'avg_train_loss': avg_loss}
        return {'train_loss': avg_loss, 'log': logs}

    def validation_step(self,
                        batch: Dict = None,
                        batch_idx: int = None) -> Dict[str, Tensor]:
        logits = self.forward(batch)
        if self.num_classes == 3:
            labels = batch['label_major']
        else:
            labels = batch['label_minor']
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        preds = torch.argmax(logits, dim=1)
        val_acc = accuracy(preds, labels.view(-1), num_classes=self.num_classes)
        val_pr = precision(preds, labels.view(-1), num_classes=self.num_classes)
        val_rc = recall(preds, labels.view(-1), num_classes=self.num_classes)

        return {'val_loss': loss,
                'val_acc': val_acc,
                'val_pr': val_pr,
                'val_rc': val_rc}

    def validation_epoch_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_pr = torch.stack([x['val_pr'] for x in outputs]).mean()
        avg_rc = torch.stack([x['val_rc'] for x in outputs]).mean()

        logs = {'avg_val_loss': avg_loss,
                'avg_val_acc': avg_acc,
                'avg_val_pr': avg_pr,
                'avg_val_rc': avg_rc}
        return {'val_loss': avg_loss, 'log': logs}

    def test_step(self,
                  batch: Dict = None,
                  batch_idx: int = None) -> Dict[str, Tensor]:
        logits = self.forward(batch)
        if self.num_classes == 3:
            labels = batch['label_major']
        else:
            labels = batch['label_minor']
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        preds = torch.argmax(logits, dim=1)
        test_acc = accuracy(preds, labels.view(-1), num_classes=self.num_classes)
        test_pr = precision(preds, labels.view(-1), num_classes=self.num_classes)
        test_rc = recall(preds, labels.view(-1), num_classes=self.num_classes)

        return {'test_loss': loss,
                'test_acc': test_acc,
                'test_pr': test_pr,
                'test_rc': test_rc}

    def test_epoch_end(
            self, outputs: Union[EvalResult, List[EvalResult]]
    ) -> Dict[str, Union[Dict[str, Any], Any]]:
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        avg_pr = torch.stack([x['test_pr'] for x in outputs]).mean()
        avg_rc = torch.stack([x['test_rc'] for x in outputs]).mean()
        logs = {'avg_test_loss': avg_loss,
                'avg_test_acc': avg_acc,
                'avg_test_pr': avg_pr,
                'avg_test_rc': avg_rc}
        return {'test_loss': avg_loss, 'log': logs}