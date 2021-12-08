import torch
import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration
from fairseq.data import FastaDataset, EncodedFastaDataset, Dictionary, BaseWrapperDataset
from constants import tokenization, neucleotides
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from omegaconf import DictConfig, OmegaConf
import hydra

import torchmetrics

from typing import List, Dict
from pytorch_lightning.loggers import WandbLogger

from pandas import DataFrame as df
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint

'''
TODOs (10/17/21):
* figure out reasonable train/valid set
* run a few baselines in this setup to get a handle on what performnace is like
* ESM-1b pretrained representations
* Alphafold
'''

class CSVDataset(Dataset):
    def __init__(self, csv_path, split, split_seed=42, supervised=True):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        # print(self.df)
        # print(self.df['seq'][0])
        if supervised:
            self.df = self.df.dropna()
        self.data = self.split(split)[['seq', 'bind']].to_dict('records')
        self.data = [x for x in self.data if x not in self.data[16*711:16*714]]
        self.data =self.data[:1]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    def split(self, split):
        if split.lower() == 'train':
            return self.df[self.df['split'] == 0]
        elif split.lower() == 'val':
            return self.df[self.df['split'] == 1]
        elif split.lower() == 'test':
            return self.df[self.df['split'] == 2]



class SupervisedRebaseDataset(BaseWrapperDataset):
    '''
    Filters a rebased dataset for entries that have supervised labels
    '''
    def __init__(self, dataset: FastaDataset):
        super().__init__(dataset)
        # print(len(dataset))
        self.filtered_indices = []

        # example desc: ['>AacAA1ORF2951P', 'GATATC', '280', 'aa']
        self.dna_element = 1 # element in desc corresponding to the DNA

        def encodes_as_dna(s: str):
            for c in s:
                if c not in list(neucleotides.keys()):
                    return False
            return True

        # filter indicies which don't have supervised labels
        for idx, (desc, seq) in enumerate(dataset):
            # if len(desc.split()) == 4 and encodes_as_dna(desc.split()[self.dna_element]):
            if len(desc.split(' ')) >= 2 and encodes_as_dna(desc.split(' ')[self.dna_element]):
                self.filtered_indices.append(idx)
        # print(len(self.dataset[0]))
        # print(self.dataset[0])
        # print('size:', len(self.filtered_indices))

    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        # translate to our filtered indices
        new_idx = self.filtered_indices[idx]
        desc, seq = self.dataset[new_idx]
        try:
            return {
                'seq': self.dataset[new_idx][1].replace(' ', ''),
                'bind': self.dataset[new_idx][0].split(' ')[self.dna_element]
            }     
        except IndexError:
            # print('hello')
            # print(self.dataset[new_idx][0].split(' '))
            # # print(self.dataset[new_idx][0])
            # print(self.dataset[new_idx])

            # print(new_idx)
            # print({
            #     'protein': self.dataset[new_idx][1].replace(' ', ''),
            #     'dna': self.dataset[new_idx][0].split(' ')[self.dna_element]
            # })
            return {
                'seq': self.dataset[new_idx][1].replace(' ', ''),
                'bind': self.dataset[new_idx][0].split(' ')[self.dna_element]
            }
            # quit()
            


class EncodedFastaDatasetWrapper(BaseWrapperDataset):
    """
    EncodedFastaDataset implemented as a wrapper
    """

    def __init__(self, dataset, dictionary, apply_bos=True, apply_eos=False):
        '''
        Options to apply bos and eos tokens.   will usually have eos already applied,
        but won't have bos. Hence the defaults here.
        '''
        super().__init__(dataset)
        self.dictionary = dictionary
        self.apply_bos = apply_bos
        self.apply_eos = apply_eos

    def __getitem__(self, idx):
        desc, seq = self.dataset[idx]
        return {
            k: self.dictionary.encode_line(v, line_tokenizer=list).long()
            for k, v in self.dataset[idx].items()
        }
    def __len__(self):
        return len(self.dataset)
    def collate_tensors(self, batch: List[torch.tensor]):
        batch_size = len(batch)
        max_len = max(el.size(0) for el in batch)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.apply_bos) + int(self.apply_eos) # eos and bos
            ),
            dtype=torch.int64,
        ).fill_(self.dictionary.pad())

        if self.apply_bos:
            tokens[:, 0] = self.dictionary.bos()

        for idx, el in enumerate(batch):
            tokens[idx, 1:(el.size(0) + 1)] = el

            if self.apply_eos:
                tokens[idx, el.size(0) + 1] = self.dictionary.eos()
        
        return tokens

    def collater(self, batch):
        if isinstance(batch, list) and torch.is_tensor(batch[0]):
            return self.collate_tensors(batch)
        else:
            return self.collate_dicts(batch)

    def collate_dicts(self, batch: List[Dict[str, torch.tensor]]):
        '''
        combine sequences of the form
        [
            {
                'key1': torch.tensor,
                'key2': torch.tensor
            },
            {
                'key1': torch.tensor,
                'key2': torch.tensor
            },
        ]
        into a collated form:
        {
            'key1': torch.tensor,
            'key2': torch.tensor,
        }
        applying the padding correctly to capture different lengths
        '''

        def select_by_key(lst: List[Dict], key):
            return [el[key] for el in lst]

        return {
            key: self.collate_tensors(
                select_by_key(batch, key)
            )
            for key in batch[0].keys()
        }
            
        
class InlineDictionary(Dictionary):
    @classmethod
    def from_list(cls, lst: List[str]):
        d = cls()
        for idx, word in enumerate(lst):
            count = len(lst) - idx
            d.add_symbol(word, n=count, overwrite=False)
        return d

class RebaseT5(pl.LightningModule):
    def __init__(self, cfg):
        super(RebaseT5, self).__init__()
        self.save_hyperparameters(cfg)
        print('batch size', self.hparams.model.batch_size)
        self.batch_size = self.hparams.model.batch_size

        self.dictionary = InlineDictionary.from_list(
            tokenization['toks']
        )
        self.cfg = cfg
        

        print(len(self.dictionary))

        t5_config=T5Config(
            vocab_size=len(self.dictionary),
            decoder_start_token_id=self.dictionary.bos(),
            # TODO: grab these from the config
            d_model=768,
            d_ff=self.hparams.model.d_ff,
            num_layers=1,
        )

        self.model = T5ForConditionalGeneration(t5_config)
        self.accuracy = torchmetrics.Accuracy()
        print('initialized')

    def training_step(self, batch, batch_idx):
        
        # input_ids, attention_mask, labels
        # torch.grad(   )
        # mask = batch['protein'].clone()
        # def func(x):
        #     if x == self.dictionary.pad():
        #         return 0
        #     return 1
        mask = (batch['seq'] != self.dictionary.pad()).int()
        
            
            
        
       
        # print(max([batch['bind'][i] for i in range(batch['bind'].shape[0])]))
        output = self.model(input_ids=batch['seq'], attention_mask=mask.to(self.device), labels=batch['bind'])
        
        # print(batch) 
        # # print(mask)
        # # # print(1 if batch['protein'] != self.dictionary.pad() else 0)
        # print(output['logits'].argmax(-1))
        # # print(self.accuracy(output['logits'].argmax(-1), batch['dna']))
        # quit()
        # log accuracy
        self.log('train_acc', self.accuracy(output['logits'].argmax(-1), batch['bind']), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', float(output.loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {
            'loss': output.loss,
            'batch_size': batch['seq'].size(0)
        }
    
    def validation_step(self, batch, batch_idx):
        # input_ids, attention_mask, labels
        mask = (batch['seq'] != self.dictionary.pad()).int()
        
            
       
        # print(max([batch['bind'][i] for i in range(batch['bind'].shape[0])]))
        output = self.model(input_ids=batch['seq'], attention_mask=mask.to(self.device), labels=batch['bind'])
        # if batch_idx == 0:
        #     print('output:', output['logits'].argmax(-1)[0], 'label:', batch['bind'][0])
        # print(batch) 
        # # print(mask)
        # # # print(1 if batch['protein'] != self.dictionary.pad() else 0)
        # print(output['logits'].argmax(-1))
        # # print(self.accuracy(output['logits'].argmax(-1), batch['dna']))
        # quit()
        # log accuracy
        if batch_idx == 0 and self.current_epoch%100 == 0:
            print('output:', output['logits'].argmax(-1)[0], 'label:', batch['bind'][0])
        self.log('val_acc', self.accuracy(output['logits'].argmax(-1), batch['bind']), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_loss', int(output.loss), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        return {
            'loss': output.loss,
            'batch_size': batch['seq'].size(0)
        }
    
    def train_dataloader(self):
        # print(self.hparams.io.train)
        dataset = EncodedFastaDatasetWrapper(
            CSVDataset(self.cfg.io.final, 'train'),

            self.dictionary
        )
        # print('length of dataset', len(dataset))

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collater)

        return dataloader
    def val_dataloader(self):
        dataset = EncodedFastaDatasetWrapper(
            CSVDataset(self.cfg.io.final, 'val'),

            self.dictionary
        )
        # print('length of dataset', len(dataset))

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collater)

        return dataloader 
    #     # print(self.hparams.io.train)
    #     dataset = EncodedFastaDatasetWrapper(
    #         SupervisedRebaseDataset(
    #             FastaDataset(self.hparams.io.val)
    #         ),
    #         self.dictionary
    #     )
    #     # print('length of dataset', len(dataset))

    #     dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collater)

    #     return dataloader

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.model.lr)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(opt, patience=150, verbose=True),
                "monitor": "val_acc",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

@hydra.main(config_path='configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    model = RebaseT5(cfg)
    max1 = 0
    # for idx, batch in enumerate(model.train_dataloader()):
    #     if batch['seq'].shape[1] >max1:
    #         max1 = batch['seq'].shape[1]
    #         maxidx = idx
    #     if idx >= 710:
    #         print(batch['seq'].shape[1], max1)
    #     if idx == 720:
    #         # print(max1)
    #         print(maxidx)

    #         import pdb; pdb.set_trace()
    wandb_logger = WandbLogger(project="Focus")
    checkpoint_callback = ModelCheckpoint(monitor="train_acc") 

    trainer = pl.Trainer(gpus=1, 
        logger=wandb_logger,
        # limit_train_batches=2,
        # limit_train_epochs=3
        # auto_scale_batch_size=True,
        callbacks=[checkpoint_callback]


        )
    trainer.tune(model)
    trainer.fit(model)
    print(max(len(batch['seq'][0]) for idx, batch in model.train_dataloader()))

if __name__ == '__main__':
    main()