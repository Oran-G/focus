import torch
import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration
from fairseq.data import FastaDataset, EncodedFastaDataset, Dictionary, BaseWrapperDataset
from constants import tokenization
from torch.utils.data import DataLoader

from omegaconf import DictConfig, OmegaConf
import hydra


from typing import List

# import fairseq.data.fairseq_dataset
# fairseq.data.fairseq_dataset.fasta_file_path = lambda x: x


'''
TODO:
* process the DNA label and amino acid sequence independently
* get labels, input_ids, attention_mask into the correct form
* make sure embedding is working properly
* make sure eos is in the right position
'''

class SupervisedRebaseDataset(BaseWrapperDataset):
    '''
    Filters a rebased dataset for entries that have supervised labels
    '''
    def __init__(self, dataset: FastaDataset):
        super().__init__(dataset)

        self.filtered_indices = []
        
        # filter indicies which don't have supervised labels
        for idx, (desc, seq) in enumerate(dataset):
            if len(desc.split()) == 4:
                self.filtered_indices.append(idx)
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        # translate to our filtered indices
        new_idx = self.filtered_indices[idx]
        return self.dataset[new_idx]

class EncodedFastaDatasetWrapper(BaseWrapperDataset):
    """
    EncodedFastaDataset implemented as a wrapper
    """

    def __init__(self, dataset, dictionary):
        super().__init__(dataset)
        self.dictionary = dictionary

    def __getitem__(self, idx):
        desc, seq = self.dataset[idx]
        return self.dictionary.encode_line(seq, line_tokenizer=list).long()


    def collater(self, batch):
        batch_size = len(batch)
        max_len = max(el.size(0) for el in batch)
        tokens = torch.empty(
            (
                batch_size,
                max_len + 2 # eos and bos
            ),
            dtype=torch.int64,
        ).fill_(self.dictionary.pad())

        tokens[:, 0] = self.dictionary.bos()
        tokens[:, -1]= self.dictionary.eos()

        for idx, el in enumerate(batch):
            tokens[idx, 1:(el.size(0) + 1)] = el
        
        return tokens
        
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
        self.cfg = cfg

        self.dictionary = InlineDictionary.from_list(
            tokenization['toks']
        )

        t5_config=T5Config(
            vocab_size=len(self.dictionary),
            decoder_start_token_id=self.dictionary.bos(),
            # TODO: grab these from the config
            d_model=768,
            d_ff=768,
            num_layers=4,
        )

        self.model = T5ForConditionalGeneration(t5_config)

    def training_step(self, batch, batch_idx):
        # input_ids, attention_mask, labels
        import pdb; pdb.set_trace()
        output = self.model(input_ids=batch, labels=batch)
        return {
            'loss': output.loss,
            'batch_size': batch['input_ids'].size(0)
        }
    
    def validation_step(self, batch):
        dataset = FastaDataset(cfg.data.protein_fasta)
        # grab validation steps?
        import pdb; pdb.set_trace()
    
    def train_dataloader(self):
        dataset = EncodedFastaDatasetWrapper(
            SupervisedRebaseDataset(
                FastaDataset(self.cfg.io.input)
            ),
            self.dictionary
        )

        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=dataset.collater)

        import pdb; pdb.set_trace()

        return dataloader

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        return opt

@hydra.main(config_path='configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    model = RebaseT5(cfg)

    trainer = pl.Trainer()
    train_dataloader = model.train_dataloader()
    trainer.fit(model, train_dataloader)

if __name__ == '__main__':
    main()