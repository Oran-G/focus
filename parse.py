import numpy as np

import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq
import argparse
if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description='Proccess REBASE into parquet')
    parser.add_argument('--input', default='~/dev/deeplearning/focus/data/All_Type_II_restriction_enzyme_genes_Protein.txt', type=str,
                        help='Path to/name of input file')
    parser.add_argument('--output', default='~/dev/deeplearning/focus/data/example.parquet', type=str,
                        help='Path to/name of output file. NEEDS TO BE ".parquet"')
    args = parser.parse_args()
    with open(args.input, 'r') as f:
        coke = f.readlines()
        df = pd.DataFrame(columns=['name', 'seq', 'bind'])
        start = False
        seq = ''
        name = ''
        bind = ''
        i = 0
        for line in coke:
            if line == '\n':
                if start ==  True:
                    # print(seq, name, bind)
                    df = df.append({'name':name, 'seq':seq, 'bind': bind if bind != '' else None}, ignore_index=True)
                    
                    seq = ''
                    name = ''
                    bind = ''
                    if i %1000 == 1:
                        print(i)
                    i+=1
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
