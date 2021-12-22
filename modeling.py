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
        # print(self.df['seq'].apply(len).mean())
        # quit()
        self.data = self.split(split)[['seq', 'bind']].to_dict('records')
        self.data = [x for x in self.data if x not in self.data[16*711:16*714]]
        self.data =self.data
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    def split(self, split):
        if split.lower() == 'train':
            # print(len(self.df[0:int(.7*len(self.df))]))
            # return self.df[0:int(.7*len(self.df))]
            return self.df[self.df['split'] == 0]
        elif split.lower() == 'val':
            # print(len(self.df[int(.7*len(self.df)):int(.85*len(self.df))]))
            
            # return self.df[int(.7*len(self.df)):int(.85*len(self.df))]
            return self.df[self.df['split'] == 1]
        elif split.lower() == 'test':
            
            # return self.df[int(.85*len(self.df)):]
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
        # desc, seq = self.dataset[idx]
        return {
            k: self.dictionary.encode_line(v, line_tokenizer=list, append_eos=False).long()
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
            tokens[idx, int(self.apply_bos):(el.size(0) + int(self.apply_bos))] = el

            # import pdb; pdb.set_trace()
            if self.apply_eos:
                tokens[idx, el.size(0) + int(self.apply_bos)] = self.dictionary.eos()
        
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
            decoder_start_token_id=self.dictionary.pad(),
            # TODO: grab these from the config
            d_model=self.hparams.model.d_model,
            d_ff=self.hparams.model.d_ff,
            num_layers=self.hparams.model.layers,
            pad_token_id=self.dictionary.pad(),
            eos_token_id=self.dictionary.eos(),
        )

        self.model = T5ForConditionalGeneration(t5_config)
        self.accuracy = torchmetrics.Accuracy(ignore_index=-100)
        self.actual_batch_size = self.hparams.model.gpu*self.hparams.model.per_gpu if self.hparams.model.gpu != 0 else 1
        print('initialized')

    def training_step(self, batch, batch_idx):
        label_mask = (batch['bind'] == self.dictionary.pad())
        batch['bind'][label_mask] = -100
        

        # import pdb; pdb.set_trace()
        # 1 for tokens that are not masked; 0 for tokens that are masked
        mask = (batch['seq'] != self.dictionary.pad()).int()

        output = self.model(input_ids=batch['seq'], attention_mask=mask, labels=batch['bind'])
        


        if batch_idx % 10000 == 0:

            print('output:', output['logits'].argmax(-1)[0], 'label:', batch['bind'][0])
            print(self.model.state_dict()['lm_head.weight'])

        # quit()
        # log accuracy
        # self.log('train_acc', self.accuracy(output['logits'].argmax(-1), batch['bind']), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', float(output.loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {
            'loss': output.loss,
            'batch_size': batch['seq'].size(0)
        }
    
    def validation_step(self, batch, batch_idx):
        label_mask = (batch['bind'] == self.dictionary.pad())
        batch['bind'][label_mask] = -100
        

        # import pdb; pdb.set_trace()
        # 1 for tokens that are not masked; 0 for tokens that are masked
        mask = (batch['seq'] != self.dictionary.pad()).int()

        output = self.model(input_ids=batch['seq'], attention_mask=mask,  labels=batch['bind'])

        if True:
            print('output:', output['logits'].argmax(-1)[0], 'label:', batch['bind'][0])
            print(self.model.state_dict()['lm_head.weight'])
        # self.log('val_acc', self.accuracy(output['logits'].argmax(-1), batch['bind']), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_loss', int(output.loss), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        return {
            'loss': output.loss,
            'batch_size': batch['seq'].size(0)
        }
    
    def train_dataloader(self):
        dataset = EncodedFastaDatasetWrapper(
            CSVDataset(self.cfg.io.final, 'train'),

            self.dictionary,
            apply_eos=True,
            apply_bos=False,
        )
        # import pdb; pdb.set_trace()

        dataloader = DataLoader(dataset, batch_size=self.actual_batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collater)

        return dataloader
    def val_dataloader(self):
        dataset = EncodedFastaDatasetWrapper(
            CSVDataset(self.cfg.io.final, 'val'),
            self.dictionary,
            apply_eos=True,
            apply_bos=False,
        )

        dataloader = DataLoader(dataset, batch_size=self.actual_batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collater)

        return dataloader 


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.model.lr)
        # return opt
        if self.hparams.model.scheduler:
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(opt, patience=self.hparams.model.lr_patience, verbose=True),
                    "monitor": "train_loss_step",
                    'interval': 'step',
                    "frequency": 1
                    # If "monitor" references validation metrics, then "frequency" should be set to a
                    # multiple of "trainer.check_val_every_n_epoch".
                },
            }
        else:
            return opt

@hydra.main(config_path='configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    model = RebaseT5(cfg)
    max1 = 0
  

    wandb_logger = WandbLogger(project="Focus")
    wandb_logger.experiment.config.update(dict(cfg.model))
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename=cfg.model.name) 
    tune_trainer = pl.Trainer(gpus=cfg.model.gpu, 
        logger=wandb_logger,
        # limit_train_batches=2,
        # limit_train_epochs=3
        # auto_scale_batch_size=True,
        callbacks=[checkpoint_callback],
        # check_val_every_n_epoch=1000,
        # max_epochs=cfg.model.max_epochs,
        default_root_dir=cfg.io.checkpoints,
        accumulate_grad_batches=1,
        precision=cfg.model.precision,
        auto_scale_batch_size="binsearch"



        )
    # quit()
    trainer.tune(model)
    # quit()
    trainer = pl.Trainer(gpus=cfg.model.gpu, 
        logger=wandb_logger,
        # limit_train_batches=2,
        # limit_train_epochs=3
        # auto_scale_batch_size=True,
        callbacks=[checkpoint_callback],
        # check_val_every_n_epoch=1000,
        # max_epochs=cfg.model.max_epochs,
        default_root_dir=cfg.io.checkpoints,
        accumulate_grad_batches=int(max(1, cfg.model.batch_size/model.hparams.batch_size)),
        precision=cfg.model.precision



        )
    trainer.fit(model)
    print(checkpoint_callback.best_model_path)
    trainer.save_checkpoint(f"{cfg.model.name}.ckpt")
if __name__ == '__main__':
    main()