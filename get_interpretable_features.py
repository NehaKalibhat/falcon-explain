import argparse
import os
import pickle 
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from torchvision.models import resnet18, resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import falcon_utils
import sys
import clip
from pytorch_grad_cam import GradCAM


_SUPPORTED_BACKBONES = [
    'resnet50',
    'resnet18'
]
_FEATURE_VECTOR_DIM = {
    'resnet50' : 2048,
    'resnet18' : 512
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str)
    parser.add_argument('--encoder_ckpt', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--probe_dataset', type=str)

    args = parser.parse_args()

    # Get model
    assert args.backbone in _SUPPORTED_BACKBONES
    backbone_model = {
        'resnet18': resnet18,
        'resnet50': resnet50,
    }[args.backbone]

    encoder = backbone_model()
    
    # Load state dict 
    print(f'Loading state - {args.encoder_ckpt}')
    state = torch.load(args.encoder_ckpt)
    encoder.load_state_dict(state)
    
    # Remove FC layer since we are interested in explaining pre-FC representation layer 
    # To explain previous layers, set subsequent layers to identity
    # Eg. for ResNet layer 2 neurons, set layer 3 and layer 4 to identity 
    encoder.fc = nn.Identity()
    encoder = encoder.cuda()
    
    # Extract activations from a probe vision dataset, eg. ImageNet
    print('Extracting all target representations from the target encoder using the probe dataset')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
    ])
    
    dataset = ImageFolder(args.probe_dataset, transform)
    batch_size = 512
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 16,
        pin_memory = True,
        drop_last = False,
    )
    
    size = len(dataloader.dataset)
    representations = torch.zeros((size, _FEATURE_VECTOR_DIM[args.backbone]))
    for i, (images, _) in enumerate(dataloader):
        with torch.no_grad():
            representations[i * batch_size: (i + 1) * batch_size] = encoder(images.cuda())
    
    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)
    torch.save(representations, f'{args.exp_name}/representations.pth.tar')
    
    representations = F.normalize(representations, dim = 1)
    
    # Get highly activating sample indices for each feature
    lim = 5 # This limit is empirically selected
    high_thresh = representations.mean() + lim * representations.std()
    print(high_thresh)

    sample_to_rep = {}

    for i in range(representations.shape[1]):
        highly_idx = torch.where(representations[:,i] >= high_thresh)[0]
        if highly_idx.shape[0] > 0:
            for idx in highly_idx:
                if idx.item() in sample_to_rep:
                    sample_to_rep[idx.item()].append(i)
                else:    
                    sample_to_rep[idx.item()] = [i]

    groups = {}
    for sam in sample_to_rep:
        s = ','.join([str(i) for i in sample_to_rep[sam]])
        if s in groups:
            groups[s].append(sam)
        else:
            groups[s] = [sam]

    print('Number of feature groups identified:', len(groups))
    groups = dict([(i, groups[i]) for i in groups if len(groups[i]) > 10])
    print('Number of feature groups with at least 10 samples:', len(groups))
        
    f = open(f'{args.exp_name}/feature_groups.pkl','wb')
    pickle.dump(groups, f)
    f.close()
    
    # Prepare grad cam and CLIP to extract interpretable feature groups
    cam = GradCAM(model = encoder,
                  target_layers = encoder.layer4, # change target_layers according to the layer we wish to explain
                  use_cuda = True) 
    
    clip_model, _ = clip.load('ViT-B/32', device='cuda')
    resize_transform = transforms.Compose([transforms.Resize((224, 224))])
    inv_norm_transform = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ],
                             std = [ 1/0.228, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406],
                             std = [ 1., 1., 1. ]),
    ])
    
    avg_clip_cos = {}
    
    if not os.path.exists(f'{args.exp_name}/highly_act_images'):
        os.makedirs(f'{args.exp_name}/highly_act_images')
    if not os.path.exists(f'{args.exp_name}/highly_act_clip_emb'):
        os.makedirs(f'{args.exp_name}/highly_act_clip_emb')   
    for feat_str in groups:
        feats = [int(i) for i in feat_str.split(',')]
                
        highly_idx = torch.LongTensor(groups[feat_str])

        highly_act_images, highly_act_clip_features, highly_act_cropped_images = falcon_utils.get_clip_image_features(clip_model = clip_model,
                                                                                                                      probe_dataset = dataset,
                                                                                                                      sample_indices = highly_idx,
                                                                                                                      batch_size = batch_size,
                                                                                                                      target_feature_group = feats,
                                                                                                                      grad_cam = cam,
                                                                                                                      resize_transform = resize_transform)
        save_image(inv_norm_transform(highly_act_cropped_images), f'{args.exp_name}/highly_act_images/{feat_str}.png')
        torch.save(highly_act_clip_features, f'{args.exp_name}/highly_act_clip_emb/{feat_str}.pth.tar')
        
        avg_clip_cos[feat_str] = torch.matmul(highly_act_clip_features, highly_act_clip_features.T).mean().cpu()

    f = open(f'{args.exp_name}/avg_clip_cos.pkl','wb')
    pickle.dump(avg_clip_cos, f)
    f.close()
    
    int_groups = dict([grp for grp in groups.items() if avg_clip_cos[grp[0]] > 0.8])
    
    print('Number of interpretable feature groups:', len(int_groups))
    
    f = open(f'{args.exp_name}/int_feature_groups.pkl','wb')
    pickle.dump(int_groups, f)
    f.close()
    
    
if __name__ == '__main__':
    main()
