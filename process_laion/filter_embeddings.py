import numpy as np

# Provide path to LAION metadata here and create captions.txt to write all captions
path = 'deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings'

unique_indices = np.load('unique_indices.npy')

new_size = 0
size = 0

start = 0
for i in range(410):
    print(i)
    batch = np.load(f'{path}/text_emb/text_emb_{i}.npy')
    end = start + batch.shape[0]
    
    batch_unique_indices = unique_indices[(unique_indices >= start) & (unique_indices < end)]
    new_size += batch_unique_indices.shape[0]
    np.save(f'{path}/text_emb_filtered/text_emb_{i}.npy', batch[batch_unique_indices - size])
    
    start = end
    size += batch.shape[0]
    
    
print('Old Size:', size)
print('New Size:', new_size)