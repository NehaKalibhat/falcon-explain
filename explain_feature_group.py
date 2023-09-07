import argparse
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import pickle
import pandas as pd
import pdb
 
import sys
import clip

import falcon_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--probe_dataset', type=str)
    parser.add_argument('--feature_group', type=str)
    
    args = parser.parse_args()
    
    # Get all saved artifacts
    print(f'Loading all saved artifacts from {args.exp_name}')
    encoder_representations = torch.load(f'{args.exp_name}/representations.pth.tar')
    encoder_representations = F.normalize(encoder_representations, dim = 1)
    f = open(f'{args.exp_name}/int_feature_groups.pkl','rb')
    feature_groups = pickle.load(f)
    f.close()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
    ])
    
    # Get image and caption datasets
    image_dataset = ImageFolder(args.probe_dataset, transform)
    caption_dataset = falcon_utils.LAION(root = 'deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings', transform = transforms.ToTensor())
    caption_dataloader = DataLoader(caption_dataset, 
                                    batch_size = 1, 
                                    shuffle = False, 
                                    num_workers = 16,
                                    pin_memory = True)

    # Threshold for lowly activating samples
    lim = 5
    low_thresh = encoder_representations.mean()
    
    # Get CLIP model
    clip_model, _ = clip.load('ViT-B/32', device='cuda')
    inv_norm_transform = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ],
                             std = [ 1/0.228, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406],
                             std = [ 1., 1., 1. ]),
    ])
    if not os.path.exists(f'{args.exp_name}/lowly_act_images'):
        os.makedirs(f'{args.exp_name}/lowly_act_images')
    if not os.path.exists(f'{args.exp_name}/captions'):
        os.makedirs(f'{args.exp_name}/captions')
        
    if not os.path.exists(f'{args.exp_name}/falcon_concepts.csv'):
        df = pd.DataFrame(columns = ['group', 'concept_set_words', 'concept_set_noun_phrases'])
        df.set_index('group')
    else:
        df = pd.read_csv(f'{args.exp_name}/falcon_concepts.csv', index_col = 'group')

    # Caption the highly and lowly activating images for given feature group
    feat_str = args.feature_group
    print('Feature group:', feat_str)

    group = torch.LongTensor([int(i) for i in feat_str.split(',')])

    highly_act_idx = torch.LongTensor(feature_groups[feat_str])

    idx_wo_feats = torch.LongTensor([i for i in torch.arange(0, encoder_representations.shape[1]).long() if i not in group])
    mean_highly_act = encoder_representations[highly_act_idx][:, idx_wo_feats].mean(0)

    lowly_act_idx = torch.where((encoder_representations[:,group] < low_thresh).all(dim = 1))[0]
    lowly_act_sorted = torch.matmul(encoder_representations[lowly_act_idx][:, idx_wo_feats], mean_highly_act).sort(descending = True)
    lowly_act_idx = lowly_act_idx[lowly_act_sorted.indices[lowly_act_sorted.values >= 0.3]][:10]

    print('Number of highly activating images: ', highly_act_idx.shape[0])
    print('Number of lowly activating images: ', lowly_act_idx.shape[0])

    highly_act_clip_emb = torch.load(f'{args.exp_name}/highly_act_clip_emb/{feat_str}.pth.tar')
    lowly_act_images, lowly_act_clip_emb, _ = falcon_utils.get_clip_image_features(clip_model = clip_model,
                                                                                   probe_dataset = image_dataset,
                                                                                   sample_indices = lowly_act_idx)

    save_image(inv_norm_transform(lowly_act_images), f'{args.exp_name}/lowly_act_images/{feat_str}.png')

    caption_set, score_set = falcon_utils.caption_images(torch.cat([highly_act_clip_emb, lowly_act_clip_emb]).to(torch.float16), caption_dataloader, caption_dataset)

    highly_act_caption_set, highly_act_score_set = caption_set[:highly_act_clip_emb.shape[0]], score_set[:highly_act_clip_emb.shape[0]]
    lowly_act_caption_set, lowly_act_score_set = ([],[]) if lowly_act_clip_emb.shape[0] == 0 else (caption_set[highly_act_clip_emb.shape[0]:], score_set[highly_act_clip_emb.shape[0]:])

    f = open(f'{args.exp_name}/captions/{feat_str}.pkl','wb')
    pickle.dump({'highly_act_caption_set':highly_act_caption_set,
                'highly_act_score_set':highly_act_score_set, 
                'lowly_act_caption_set': lowly_act_caption_set, 
                'lowly_act_score_set':lowly_act_score_set}, f)
    f.close()

    # Parse captions and do contrastive concept extraction
    all_noun_phrases, all_words, _, _, _ = falcon_utils.parse_captions(highly_act_caption_set, highly_act_score_set)
    lowly_words = []
    if len(lowly_act_caption_set) > 0:
        noun_phrases, words, _, _, _ = falcon_utils.parse_captions(lowly_act_caption_set, lowly_act_score_set)

        thresh = 0.1
        noun_phrases = [i[0] for i in noun_phrases if i[1] >= thresh]
        words = [i[0] for i in words if i[1] >= thresh]

        all_noun_phrases = [i for i in all_noun_phrases if i[0] not in noun_phrases]
        all_words = [i for i in all_words if i[0] not in words]

    # thresh = 0.08
    noun_phrases = [i for i in all_noun_phrases]
    words = [i for i in all_words]

    word_set = sorted(words, key = lambda k : k[1], reverse = True)[:8]
    phrases_set = sorted(noun_phrases, key = lambda k : k[1], reverse = True)[:8]

    df.loc[feat_str] = [feat_str, word_set, phrases_set]

    df.to_csv(f'{args.exp_name}/falcon_concepts.csv')

        
if __name__ == '__main__':
    main()
