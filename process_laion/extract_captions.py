import dask.dataframe as dd
import pandas as pd
import numpy as np

# Provide path to LAION metadata here and create captions.txt to write all captions
path = 'deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings'
offset = 0
unique_captions = set()
unique_indices = []
    
with open(f'{path}/metadata/captions.txt', 'a') as f:
    for i in range(410):
        print(i)
        df_idx = pd.read_parquet(f'{path}/metadata/metadata_{i}.parquet')
        batch_captions = df_idx.caption
        for c in range(len(batch_captions)):
            cap = batch_captions[c].rstrip()
            if cap not in unique_captions:
                unique_indices.append(c + offset)
                unique_captions.add(cap)
                
                f.write(cap + '\n')
                
        
        offset += len(batch_captions)
        
print('Number of unique captions:', len(unique_indices))
np.save('laion_unique_indices.npy', np.array(unique_indices))