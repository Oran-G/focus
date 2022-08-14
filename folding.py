# import pdb; pdb.set_trace()
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from fairseq.data import FastaDataset, EncodedFastaDataset, Dictionary, BaseWrapperDataset
from constants import tokenization, neucleotides
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import pdb; pdb.set_trace()
from omegaconf import DictConfig, OmegaConf
import hydra
# import pdb; pdb.set_trace()
import torchmetrics
# import pdb; pdb.set_trace()
from typing import List, Dict
from pytorch_lightning.loggers import WandbLogger
# import pdb; pdb.set_trace()

from pandas import DataFrame as df
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# import torch
import pandas as pd
# import pdb; pdb.set_trace()

import esm.inverse_folding
import esm
import torch_geometric
# import pdb; pdb.set_trace()
from GPUtil import showUtilization as gpu_usage

import time
from pl_bolts.datamodules.async_dataloader import AsynchronousLoader
import os
import json
import wandb
'''
TODOs (10/17/21):
* figure out reasonable train/valid set
* run a few baselines in this setup to get a handle on what performnace is like
* ESM-1b pretrained representations
* Alphafold
'''

class CSVDataset(Dataset):
    def __init__(self, csv_path, split, split_seed=42, supervised=True,plddt=85):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        # print(self.df)
        # print(self.df['seq'][0])
        if supervised:
            self.df = self.df.dropna()
        # print(self.df['seq'].apply(len).mean())
        # quit()
        def cat(x):
            return (x[:1023] if len(x) > 1024 else x)
        self.df['seq'] = self.df['seq'].apply(cat)
        print("pre alpha",len(self.df))
        def alpha(ids):
            return os.path.isfile(f'/vast/og2114/rebase/20220519/output/{ids}/ranked_0.pdb')
        self.df  = self.df[self.df['id'].apply(alpha) ==True ]
        print("after alpha",len(self.df))
        print("pre beta",len(self.df))
        def beta(ids):
            return  max(json.load(open(f'/vast/og2114/rebase/20220519/output/{ids}/ranking_debug.json'))['plddts'].values()) >= plddt
        self.df  = self.df[self.df['id'].apply(beta) ==True ]
        print("after beta",len(self.df))

        self.data = self.split(split)[['seq','bind', 'id']].to_dict('records')
    
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
            tmp = self.df[self.df['split'] != 1]
            return tmp[tmp['split'] != 2]
        
        elif split.lower() == 'val':
            # print(len(self.df[int(.7*len(self.df)):int(.85*len(self.df))]))
            
            # return self.df[int(.7*len(self.df)):int(.85*len(self.df))]
            return self.df[self.df['split'] == 1]
        elif split.lower() == 'test':
            
            # return self.df[int(.85*len(self.df)):]
            return self.df[self.df['split'] == 2]



# class SupervisedRebaseDataset(BaseWrapperDataset):
#     '''
#     Filters a rebased dataset for entries that have supervised labels
#     '''
#     def __init__(self, dataset: FastaDataset):
#         super().__init__(dataset)
#         # print(len(dataset))
#         self.filtered_indices = []

#         # example desc: ['>AacAA1ORF2951P', 'GATATC', '280', 'aa']
#         self.dna_element = 1 # element in desc corresponding to the DNA

#         def encodes_as_dna(s: str):
#             for c in s:
#                 if c not in list(neucleotides.keys()):
#                     return False
#             return True

#         # filter indicies which don't have supervised labels
#         for idx, (desc, seq) in enumerate(dataset):
#             # if len(desc.split()) == 4 and encodes_as_dna(desc.split()[self.dna_element]):
#             if len(desc.split(' ')) >= 2 and encodes_as_dna(desc.split(' ')[self.dna_element]):
#                 self.filtered_indices.append(idx)
#         # print(len(self.dataset[0]))
#         # print(self.dataset[0])
#         # print('size:', len(self.filtered_indices))

    
#     def __len__(self):
#         return len(self.filtered_indices)
    
#     def __getitem__(self, idx):
#         # translate to our filtered indices
#         new_idx = self.filtered_indices[idx]
#         desc, seq = self.dataset[new_idx]
#         try:
#             return {
#                 'seq': self.dataset[new_idx][1].replace(' ', ''),
#                 'bind': self.dataset[new_idx][0].split(' ')[self.dna_element]
#                 'name'
#             }     
#         except IndexError:
#             # print(new_idx)
#             # print({
#             #     'protein': self.dataset[new_idx][1].replace(' ', ''),
#             #     'dna': self.dataset[new_idx][0].split(' ')[self.dna_element]
#             # })
#             return {
#                 'seq': self.dataset[new_idx][1].replace(' ', ''),
#                 'bind': self.dataset[new_idx][0].split(' ')[self.dna_element]
#             }
#             # quit()
            


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
        self.ifalphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()[1]
    def __getitem__(self, idx):
        # desc, seq = self.dataset[idx]
        
        structure = esm.inverse_folding.util.load_structure(f"/vast/og2114/rebase/20220519/output/{self.dataset[idx]['id']}/ranked_0.pdb", 'A')

        coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
        #print(seq)
        # print(self.dictionary.encode_line(self.dataset[idx]['bind'], line_tokenizer=list, append_eos=False, add_if_not_exist=False).long())
        # print(coords,seq)
        # print(type(coords), type(seq))
        return {
            # 'seq': self.dictionary.encode_line(self.dataset[idx]['seq'], line_tokenizer=list, append_eos=False, add_if_not_exist=False).long(),
            # 'bind': self.dictionary.encode_line(self.dataset[idx]['bind'], line_tokenizer=list, append_eos=False, add_if_not_exist=False).long(),
            'bind':torch.tensor( self.ifalphabet.encode(self.dataset[idx]['bind'])),
            'coords': coords,
            # 'seq': self.dictionary.encode_line(seq, line_tokenizer=list, append_eos=False, add_if_not_exist=False).long()
            'seq':torch.tensor(self.ifalphabet.encode(seq))
        }

    def __len__(self):
        return len(self.dataset)
    def collate_tensors(self, batch: List[torch.tensor]):
        # import pdb; pdb.set_trace()
        #print(batch)
        #quit()
        if batch[0].ndim <= 2:
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
        else:
            bts = esm.inverse_folding.util.CoordBatchConverter(self.ifalphabet)
            return bts.from_lists(batch)

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
        # print('collate')
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

def accuracy(predict:torch.tensor, label:torch.tensor, mask:torch.tensor):
    first = (predict==label).int()
    second = first*mask
    return second.sum()/mask.sum()

class RebaseT5(pl.LightningModule):
    def __init__(self, cfg):
        super(RebaseT5, self).__init__()
        self.save_hyperparameters(cfg)
        # self.save_hyperparameters()
        print("Argument hparams: ", self.hparams)
        # needed hparams for non-lightning pre-trained weights
        print('batch size', self.hparams.model.batch_size)
        self.batch_size = self.hparams.model.batch_size
        

        self.dictionary = InlineDictionary.from_list(
            tokenization['toks']
        )
        self.cfg = cfg

        self.perplex = torch.nn.CrossEntropyLoss(reduction='none')
        

        # self.esm, self.esm_dictionary = torch.hub.load("facebookresearch/esm:main", self.hparams.esm.path)
        # self.
       
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
        self.accuracy = torchmetrics.Accuracy(ignore_index=self.dictionary.pad())
        
        self.ifmodel, self.ifalphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        
        # model = model.eval()
        # self.actual_batch_size = self.hparams.model.gpu*self.hparams.model.per_gpu if self.hparams.model.gpu != 0 else 1
        self.test_data = []
        print('initialized')
        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.NLLLoss()

    def training_step(self, batch, batch_idx):
        if self.global_step  != 0:
            #print('lr')
            self.lr_schedulers().step()
        self.ifmodel.train()
        label_mask = (batch['bind'] == self.ifalphabet.padding_idx)
        batch['bind'][label_mask] = -100
        

        #import pdb; pdb.set_trace()
        # 1 for tokens that are not masked; 0 for tokens that are masked
        start_time  = time.time()
        mask = (batch['seq'] != self.ifalphabet.padding_idx).int()


        # load ESM-1b in __init__(...)
        # convert sequence into the ESM-1b vocabulary
        # # # load up ESM-1b alphabet; convert sequences using our dictionary and ESM-1b dictionary, check that you get same ouputs
        # # # if not, write a conversion function convert(t: torch.tensor) -> torch.tensor
        # take the converted sequence, pass it through ESM-1b, get hidden representations from layer 33
        # these representations will be of shape torch.tensor [batch, seq_len, 768]
        # make sure you don't take gradients through ESM-1b; do torch.no_grad()
        # alternatively, you can do this in __init__: [for parameter in self.esm1b_model.paramters(): parmater.requires_grad=False]
        # pass that ESM-1b hidden states into self.model(..., encoder_outputs=...)
        # print(torch.cuda.memory_summary(device=None, abbreviated=True))
        #import pdb; pdb.set_trace()
        # pred = self.ifmodel.sample(batch['coords'], temprature=1)#.logits
        torch.cuda.empty_cache()
        if self.hparams.esm.esmgrad == False:
            with torch.no_grad():
                token_representations = self.ifmodel(batch['coords'][0],
                    confidence=batch['coords'][1], padding_mask=batch['coords'][4],
                    prev_output_tokens=batch['coords'][3])[1]['inner_states'][-1]#.logits
        else:
            token_representations = self.ifmodel(batch['coords'][0],
                confidence=batch['coords'][1], padding_mask=batch['coords'][4],
                prev_output_tokens=batch['coords'][3])[1]['inner_states'][-1]#.logits
        pred = self.model(encoder_outputs=[torch.transpose(token_representations, 0, 1)], attention_mask=mask, labels=batch['bind'])
        
        #loss = self.loss(pred, batch['bind'])
        if True:
            bm = ''
            for i in range(pred[1].shape[1] - batch['bind'].shape[1] - 1):
                bm+="<mask>"
            #print(pred.shape[2] - batch['bind'].shape[1])
            #bm.append("<mask>" for _ in range(pred.shape[1] - batch['bind'].shape[1]))
            #bind_mask = torch.tensor(self.ifalphabet.encode(["<mask>" for _ in range(pred.shape[2] - batch['bind'].shape[1])]))
            bind_mask = torch.tensor(self.ifalphabet.encode(bm)).to(self.device)
            binds = []
            for i in range(batch['bind'].shape[0]):
                binds.append(torch.cat((batch['bind'][i], bind_mask)))
            bind = self.dataset.collate_tensors(binds)
        #print(bind)
        #print(bind.shape)
        bind = bind.to(self.device)
        #import pdb; pdb.set_trace()
        #loss = self.loss(torch.nn.functional.log_softmax(torch.transpose(pred[1].to(self.device), 1, 2), dim=1), bind.to(self.device))
        #print((bind != self.ifalphabet.mask_idx).int())
        #print(float(accuracy(pred.argmax(-2), bind, (bind != self.ifalphabet.mask_idx).int().to(self.device))))
        #quit()
       # gpu_usage()         
        

        
        import pdb; pdb.set_trace()
        confs = self.conf(nn.functional.softmax(pred[1], dim=-1))
        self.log('top_conf', float(confs[0]), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('low_conf', float(confs[1]), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('mean_conf', float(confs[2]), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', float(pred[0]), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc',float(accuracy(nn.functional.softmax(pred[1],dim=-1).argmax(-1), batch['bind'], (batch['bind'] != self.ifalphabet.pad_idx).int().to(self.device))), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('length', int(pred[1].shape[-2]),  on_step=True,  logger=True)
        self.log('train_time', time.time()- start_time, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #import pdb; pdb.set_trace()
        # self.log('train_acc',float(accuracy(pred.argmax(-1), batch['bind'], (batch['bind'] != -100).int())), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log('train_perplex',float(self.perplexity(output['logits'], batch['bind'])), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        return {
            'loss': pred[0],
            'batch_size': batch['seq'].size(0)
        }
    
    def validation_step(self, batch, batch_idx):
        #print('start of validation loop')
        # label_mask = (batch['bind'] == self.dictionary.pad())
        label_mask = (batch['bind'] == self.ifalphabet.padding_idx)
        # batch['bind'][label_mask] = -100
        #import pdb; pdb.set_trace()
        start_time = time.time()
        mask = (batch['seq'] != self.ifalphabet.padding_idx).int()
        # pred = self.ifmodel.sample(batch['coords'], temprature=1)#.logits
        torch.cuda.empty_cache()
        #pred = self.ifmodel(batch['coords'][0], confidence=batch['coords'][1], padding_mask=batch['coords'][4], prev_output_tokens=batch['coords'][3])[0]#.logits
        token_representations = self.ifmodel(batch['coords'][0],
                confidence=batch['coords'][1], padding_mask=batch['coords'][4],
                prev_output_tokens=batch['coords'][3])[1]['inner_states'][-1]#.logits
        #import pdb; pdb.set_trace()
        pred = self.model(encoder_outputs=[torch.transpose(token_representations, 0, 1)], attention_mask=mask, labels=batch['bind'])
        bind = batch['bind']
        #if pred.shape[2] > batch['bind'].shape[1]:
        #print(pred.shape, bind.shape)
        #print(self.device)
        if True:
            bm = ''
            for i in range(pred[1].shape[1] - batch['bind'].shape[1] - 1):
                bm+="<mask>"
            #print(pred.shape[2] - batch['bind'].shape[1])
            #bm.append("<mask>" for _ in range(pred.shape[1] - batch['bind'].shape[1]))
            #bind_mask = torch.tensor(self.ifalphabet.encode(["<mask>" for _ in range(pred.shape[2] - batch['bind'].shape[1])]))
            bind_mask = torch.tensor(self.ifalphabet.encode(bm)).to(self.device)
            binds = []
            for i in range(batch['bind'].shape[0]):
                binds.append(torch.cat((batch['bind'][i], bind_mask)))
            bind = self.dataset.collate_tensors(binds)
        #print(bind)
        #print(bind.shape)
        bind = bind.to(self.device)
        #loss = self.loss(torch.nn.functional.log_softmax(torch.transpose(pred[1].to(self.device), 1, 2), dim=1), bind.to(self.device))
        #print((bind != self.ifalphabet.mask_idx).int())
        #print(float(accuracy(pred.argmax(-2), bind, (bind != self.ifalphabet.mask_idx).int().to(self.device))))
        #quit()
        # gpu_usage()
        
        # if True:
        #     print('output:', output['logits'].argmax(-1)[0], 'label:', batch['bind'][0])
        #     print(self.model.state_dict()['lm_head.weight'])
        # bind_accuracy = batch['bind'].detach()
        # bind_accuracy[label_mask] = self.dictionary.pad()
        # self.log('val_acc', self.accuracy(output['logits'].argmax(-1), bind_accuracy), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # import pdb; pdb.set_trace()
        
        for i in range(batch['seq'].shape[0]):
            
            try:
                #import pdb; pdb.set_trace()
                print((pred[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0])
                lastidx = -1 if len((pred[1].argmax(-1)[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0]) == 0 else (pred[1].argmax(-1)[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0].tolist()[0]
                print(lastidx)
                #import pdb; pdb.set_trace()
                print(pred[1].argmax(-1).tolist()[:lastidx][0])
                self.test_data.append({'seq': self.decode(batch['seq'][i].tolist()).split("<eos>")[0],
                    'bind': self.decode(batch['bind'][i].tolist()[:batch['bind'][i].tolist().index(2)]),
                    'predicted': self.decode(pred[1].argmax(-1).tolist()[:lastidx][0])})
            except IndexError:
                print('Index Error')
        import pdb; pdb.set_trace()
        confs = self.conf(nn.functional.softmax(pred[1], dim=-1))
        self.log('val_top_conf', float(confs[0]), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_low_conf', float(confs[1]), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mean_conf', float(confs[2]), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.log('val_loss', float(pred[0]), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_acc', float(accuracy(nn.functional.softmax(pred[1],dim=-1).argmax(-1), batch['bind'], (batch['bind'] != self.ifalphabet.pad_idx).int().to(self.device))), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log('val_acc', float(accuracy(pred.argmax(-1), batch['bind'], (batch['bind'] != self.dictionary.pad()).int())), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log('val_perplex',float(self.perplexity(output['logits'], batch['bind'])), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_time', time.time()- start_time, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {
            'loss': pred[0],
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

        dataloader = AsynchronousLoader(DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, collate_fn=dataset.collater), device=self.device)
        # print('train loader')
        # gpu_usage()
        return dataloader
    def val_dataloader(self):
        dataset = EncodedFastaDatasetWrapper(
            CSVDataset(self.cfg.io.final, 'val'),
            self.dictionary,
            apply_eos=True,
            apply_bos=False,
        )
        # print('val loader')
        self.dataset = dataset        
        dataloader = AsynchronousLoader(DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collater), device=self.device)
        # gpu_usage()
        # print('hi')
        return dataloader 

    def configure_optimizers(self):
        opt = torch.optim.AdamW([
                {'params': self.ifmodel.parameters()},  
                {'params': self.model.parameters()}
                ], 
            lr=float(self.hparams.model.lr))
        # return opt

        # figure out reaosnable number of total steps
        # 100k steps
        # 4% warmup to peak lr
        # linear decay after that

        if self.hparams.model.scheduler:
            return {
                'optimizer': opt,
                'lr_scheduler': get_linear_schedule_with_warmup(
                    optimizer=opt,
                    num_training_steps=400000,
                    num_warmup_steps=6000,
                )
            }
            # return {
            #     "optimizer": opt,
            #     "lr_scheduler": {
            #         "scheduler": ReduceLROnPlateau(opt, patience=self.hparams.model.lr_patience, verbose=True),
            #         "monitor": "train_loss_step",
            #         'interval': 'step',
            #         "frequency": 1
            #         # If "monitor" references validation metrics, then "frequency" should be set to a
            #         # multiple of "trainer.check_val_every_n_epoch".
            #     },
            # }
        else:
            return opt
    
    def test_step(self, batch, batch_idx):
        if True:
            print(batch['bind'])
            label_mask = (batch['bind'] == self.dictionary.pad())
            batch['bind'][label_mask] = -100

            torch.cuda.empty_cache()
            pred = self.ifmodel(batch['coords'][0], confidence=batch['coords'][1], padding_mask=batch['coords'][4], prev_output_tokens=batch['coords'][3])[0]
            # import pdb; pdb.set_trace()
            # 1 for tokens that are not masked; 0 for tokens that are masked
            mask = (batch['seq'] != self.dictionary.pad()).int()

            for i in range(batch['seq'].shape[0]):
                try:
                    print((pred[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0])
                    lastidx = -1 if len((pred.argmax(-2)[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0]) == 0 else (pred.argmax(-2)[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0].tolist()[0]
                    print(lastidx)
                    #import pdb; pdb.set_trace()
                
                    self.test_data.append({'seq': self.decode(batch['seq'][i].tolist()).split("<eos>")[0],
                        'bind': self.decode(batch['bind'][i].tolist()[:batch['bind'][i].tolist().index(2)]),
                        'predicted': self.decode(pred.argmax(-2)[i].tolist()[:lastidx])})
                except IndexError:
                    print('Index Error')
    def val_test(self):
        alls = []
        for idx, batch in iter(self.val_dataloader()):
            print(batch['bind'])
            label_mask = (batch['bind'] == self.dictionary.pad())
            batch['bind'][label_mask] = -100
            
            torch.cuda.empty_cache()
            pred = self.ifmodel(batch['coords'][0], confidence=batch['coords'][1], padding_mask=batch['coords'][4], prev_output_tokens=batch['coords'][3])[0]
            # import pdb; pdb.set_trace()
            # 1 for tokens that are not masked; 0 for tokens that are masked
            mask = (batch['seq'] != self.dictionary.pad()).int()
            
            for i in range(batch['seq'].shape[0]):
                print((pred[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0])
                lastidx = -1 if len((pred[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)) == 0 else (pred[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0]
                print(lastidx)
                import pdb; pdb.set_trace()
                alls.append({'seq': self.decode(batch['seq'][i].tolist()).split("<eos>")[0], 
                    'bind': self.decode(batch['bind'][i].tolist()[:batch['bind'][i].tolist().index(2)]), 
                    'predicted': self.decode(pred.argmax(-1)[i].tolist()[:lastidx])})
        return alls
    def decode(self, seq):
        # input -> [list] type token representation of sequence to be decoded
        #output -> [string] of sequence decoded 
        newseq = ''
        for tok in seq:
            newseq += str(self.ifalphabet.get_tok(tok))
        return newseq
    
    def conf(self, tens, target):
        h1 = []
        h2 = []
        h3 = []
        for i in range(tens.shape[0]):
            lastidx = -1 if len((tens.argmax(-1)[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0]) == 0 else (tens.argmax(-1)[i]  == self.ifalphabet.eos_idx).nonzero(as_tuple=True)[0].tolist()[0]
            highs = (torch.amax(tens[i], -1)[:lastidx]* ((tens[i].argmax(-1)[:lastidx]==target[i][:lastidx]).int())).tolist()
            h1.append(max(highs))
            h2.append(min(highs))
            h3.append((sum(highs)/len(highs)))
        return max(h1), min(h2), (sum(h3)/len(h3))


    def training_epoch_end(self, training_step_outputs):
        #if self.trainer.current_epoch % 5 == 0:
        if True:
            #self.trainer.test(self, dataloaders=self.val_dataloader())
            df1 = pd.DataFrame(self.test_data)
            import csv
            dictionaries=self.test_data
            keys = dictionaries[0].keys()
            a_file = open(f"/scratch/og2114/rebase/logs/slurm_{str(os.environ.get('SLURM_JOB_ID'))}/{self.trainer.current_epoch}-output.csv", "w")
            dict_writer = csv.DictWriter(a_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(dictionaries)
            print('hello')
            #tbl = wandb.Table(dataframe=df1)
            #wandb.log({'validation_table': tbl})
@hydra.main(version_base="1.2.0",config_path='configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    model = RebaseT5(cfg)
    # print('init')
    # checkpoint = torch.load('/scratch/og2114/rebase/logs/Focus/21hjudcf/checkpoints/both_dff-128_dmodel-768_lr-0.001_batch-512.ckpt')
    # print(checkpoint.keys())
    # model = RebaseT5.load_from_checkpoint(checkpoint_path="/scratch/og2114/rebase/logs/Focus/ufa553zz/checkpoints/esm12_both_grade3_dff-128_dmodel-768_lr-0.001_batch-512.ckpt")
    # model = RebaseT5.load_from_checkpoint(checkpoint_path='/scratch/og2114/rebase/logs/Focus/21hjudcf/checkpoints/both_dff-128_dmodel-768_lr-0.001_batch-512.ckpt')
    gpu = cfg.model.gpu
    cfg = model.hparams
    cfg.model.gpu = gpu
    wandb_logger = WandbLogger(project="Focus",save_dir=cfg.io.wandb_dir)
    # wandb_logger.experiment.config.update(dict(cfg.model))
    checkpoint_callback = ModelCheckpoint(monitor="val_loss_epoch", filename=f'{cfg.model.name}_dff-{cfg.model.d_ff}_dmodel-{cfg.model.d_model}_lr-{cfg.model.lr}_batch-{cfg.model.batch_size}', verbose=True,save_top_k=5) 
    acc_callback = ModelCheckpoint(monitor="val_acc_epoch", filename=f'acc-{cfg.model.name}_dff-{cfg.model.d_ff}_dmodel-{cfg.model.d_model}_lr-{cfg.model.lr}_batch-{cfg.model.batch_size}', verbose=True, save_top_k=5) 
    lr_monitor = LearningRateMonitor(logging_interval='step')
    print(model.batch_size)
    print('tune: ')
    # trainer.tune(model)
    model.batch_size = 8
    if int(cfg.esm.layers) == 12:
        model.batch_size = 2
    if int(cfg.esm.layers) == 34:
        model.batch_size = 1

    #model.batch_size = 2
    print(model.batch_size)
    print(int(max(1, cfg.model.batch_size/model.batch_size)))
    trainer = pl.Trainer(
        gpus=int(cfg.model.gpu), 
        logger=wandb_logger,
        # limit_train_batches=2,
        # limit_train_epochs=3
        # auto_scale_batch_size=True,
        callbacks=[checkpoint_callback, lr_monitor, acc_callback],
        # check_val_every_n_epoch=1000,
        # max_epochs=cfg.model.max_epochs,
        default_root_dir=cfg.io.checkpoints,
        accumulate_grad_batches=int(max(1, 64/model.batch_size/int(1))),
        accumulate_grad_batches=4,
        precision=cfg.model.precision,
        strategy='ddp',
        #accelerator='cpu',
        log_every_n_steps=5,
        progress_bar_refresh_rate=100,
        max_epochs=92,
        #limit_train_batches=5,
        auto_scale_batch_size="power",
        gradient_clip_val=0.5,
        )
    trainer.fit(model)
    trainer.test(model, dataloaders=model.val_dataloader())
    import csv
    dictionaries=model.test_data
    keys = dictionaries[0].keys()
    a_file = open("output.csv", "w")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(dictionaries)
    a_file.close()
    df1 = pd.DataFrame(dictionaries)
    tbl = wandb.Table(dataframe=df1)
    wandb.log({'final_table': tbl})
    #orangeisstupid
    

if __name__ == '__main__':
    main()
