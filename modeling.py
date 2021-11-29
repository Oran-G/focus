import torch
import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration
from fairseq.data import FastaDataset, EncodedFastaDataset, Dictionary, BaseWrapperDataset
from constants import tokenization, neucleotides
from torch.utils.data import DataLoader, Dataset

from omegaconf import DictConfig, OmegaConf
import hydra

import torchmetrics

from typing import List, Dict
from pytorch_lightning.loggers import WandbLogger

from pandas import DataFrame as df
import pandas as pd

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
        print(self.df)
        print(self.df['seq'][0])
        if supervised:
            self.df = self.df.dropna()
        self.data = self.split(split)[['seq', 'bind']].to_dict('records')
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.df)
    
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
        print(len(self.dataset[0]))
        print(self.dataset[0])
        print('size:', len(self.filtered_indices))

    
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
                
        
        # mask = mask.to('cpu').apply_(func).to(self.device)
        # print(mask)
        # quit()
        # masks =  mask.clone()
        # print(mask)
        # print(mask.shape)
        # for q in range(mask.shape[0]):
        #     for i in range(mask.shape[1]):
        #         # for j in range(mask.shape[2]):
        #         if mask[q][i] == True:
        #             # print(mask[q][i])
        #             mask[q][i] = 0
        #         else:
        #             mask[q][i] = 0
        # mask[0][0] = 1
        # # mask = mask[mask==True] = 1
        # # mask = mask[mask==False] = 0
        # print(mask)
        # quit()
        # print(mask)
        output = self.model(input_ids=batch['seq'], attention_mask=mask, labels=batch['bind'])
        # print(batch) 
        # # print(mask)
        # # # print(1 if batch['protein'] != self.dictionary.pad() else 0)
        # print(output['logits'].argmax(-1))
        # # print(self.accuracy(output['logits'].argmax(-1), batch['dna']))
        # quit()
        # log accuracy
        self.log('train_acc_step', self.accuracy(output['logits'].argmax(-1), batch['bind']), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {
            'loss': output.loss,
            'batch_size': batch['seq'].size(0)
        }
    
    def validation_step(self, batch, batch_idx):
        # input_ids, attention_mask, labels
        output = self.model(input_ids=batch['seq'], labels=batch['bind'])
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

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collater)

        return dataloader
    # def val_dataloader(self):
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
        return opt

@hydra.main(config_path='configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    model = RebaseT5(cfg)
    wandb_logger = WandbLogger(project="Focus")

    trainer = pl.Trainer(gpus=0, 
        logger=wandb_logger,
        # limit_train_batches=2,
        # limit_train_epochs=3


        )
    trainer.tune(model)
    trainer.fit(model)

if __name__ == '__main__':
    main()