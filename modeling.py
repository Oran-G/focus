import torch
import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration
from fairseq.data import FastaDataset, EncodedFastaDataset, Dictionary, BaseWrapperDataset
from constants import tokenization
from torch.utils.data import DataLoader

from omegaconf import DictConfig, OmegaConf
import hydra

import torchmetrics

from typing import List, Dict

'''
TODOs (10/17/21):
* figure out reasonable train/valid set
* run a few baselines in this setup to get a handle on what performnace is like
* ESM-1b pretrained representations
* Alphafold
'''

class SupervisedRebaseDataset(BaseWrapperDataset):
    '''
    Filters a rebased dataset for entries that have supervised labels
    '''
    def __init__(self, dataset: FastaDataset):
        super().__init__(dataset)

        self.filtered_indices = []

        # example desc: ['>AacAA1ORF2951P', 'GATATC', '280', 'aa']
        self.dna_element = 1 # element in desc corresponding to the DNA

        def encodes_as_dna(s: str):
            for c in s:
                if c not in ['A', 'T', 'C', 'G']:
                    return False
            return True

        # filter indicies which don't have supervised labels
        for idx, (desc, seq) in enumerate(dataset):
            if len(desc.split()) == 4 and encodes_as_dna(desc.split()[self.dna_element]):
                self.filtered_indices.append(idx)
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        # translate to our filtered indices
        new_idx = self.filtered_indices[idx]
        desc, seq = self.dataset[new_idx]

        return {
            'protein': seq.replace(' ', ''),
            'dna': desc.split()[self.dna_element]
        }        
        

class EncodedFastaDatasetWrapper(BaseWrapperDataset):
    """
    EncodedFastaDataset implemented as a wrapper
    """

    def __init__(self, dataset, dictionary, apply_bos=True, apply_eos=False):
        '''
        Options to apply bos and eos tokens. Fairseq datasets will usually have eos already applied,
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

        print(len(self.dictionary))

        t5_config=T5Config(
            vocab_size=len(self.dictionary),
            decoder_start_token_id=self.dictionary.bos(),
            # TODO: grab these from the config
            d_model=768,
            d_ff=768,
            num_layers=4,
        )

        self.model = T5ForConditionalGeneration(t5_config)
        self.accuracy = torchmetrics.Accuracy()

    def training_step(self, batch, batch_idx):
        # input_ids, attention_mask, labels
        output = self.model(input_ids=batch['protein'], labels=batch['dna'])

        # log accuracy
        self.log('train_acc_step', self.accuracy(output['logits'].argmax(-1), batch['dna']), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {
            'loss': output.loss,
            'batch_size': batch['protein'].size(0)
        }
    
    def validation_step(self, batch, batch_idx):
        # input_ids, attention_mask, labels
        output = self.model(input_ids=batch['protein'], labels=batch['dna'])
        return {
            'loss': output.loss,
            'batch_size': batch['protein'].size(0)
        }
    
    def train_dataloader(self):
        dataset = EncodedFastaDatasetWrapper(
            SupervisedRebaseDataset(
                FastaDataset(self.hparams.io.input)
            ),
            self.dictionary
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=10, collate_fn=dataset.collater)

        return dataloader

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        return opt

@hydra.main(config_path='configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    model = RebaseT5(cfg)

    trainer = pl.Trainer(gpus=-1)
    trainer.tune(model)
    trainer.fit(model)

if __name__ == '__main__':
    main()