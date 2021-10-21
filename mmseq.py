from Bio import SeqIO
import numpy as np

import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq
import argparse

from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os
# import fasta 

@hydra.main(config_path='configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    '''
    https://github.com/soedinglab/MMseqs2
    ^^ git with documentation^^
    '''
    # os.system('python3 fasta.py')
    print('python3 fasta.py')
    # os.system('brew install mmseqs2')
    print('brew install mmseqs2')
    os.system(f'mmseqs createdb {cfg.io.fasta} DB')
    print(f'mmseqs createdb {cfg.io.fasta} DB')
    os.system('mmseqs cluster DB DB_clu tmp')
    print('mmseqs cluster DB DB_clu tmp')
    os.system(f'mmseqs createtsv DB DB DB_clu {cfg.io.cluster}')
    print(f'mmseqs createtsv DB DB DB_clu {cfg.io.cluster}')


if __name__ == "__main__":
    main()