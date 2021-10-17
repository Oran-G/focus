import numpy as np

import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq
import argparse

from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm

@hydra.main(config_path='configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    with open(cfg.io.input, 'r') as f:
        coke = f.readlines()
        df = pd.DataFrame(columns=['name', 'seq', 'bind'])
        start = False
        seq, name, bind = '', '', ''
        for line in tqdm(coke):
            if line == '\n':
                if start ==  True:
                    # print(seq, name, bind)
                    df = df.append({'name':name, 'seq':seq, 'bind': bind if bind != '' else None}, ignore_index=True)
                    
                    seq = ''
                    name = ''
                    bind = ''
                    start = False
                    
            elif line[0] == '>':
                l = line.split(' ')
                name = l[0].replace('>','')
                bind = l[3]
                start = True
                # print(line.split(' '))
                # if bind >5:
                #     quit()
                # bind += 1
            else:
                seq+=line.replace(' ','').replace('\n','')
                
        table = pa.Table.from_pandas(df)
        pq.write_table(table, 'args.output')
        f.close() 


if __name__ == "__main__":
    main()