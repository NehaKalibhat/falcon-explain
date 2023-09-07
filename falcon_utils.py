import torch 
from torch.utils.data import Dataset
import torchvision
import numpy as np
import matplotlib.pyplot as plt 
import linecache
import pdb

from textblob import TextBlob, Word
import re 
from nltk.corpus import stopwords


_DISCARD_WORDS = ['photo', 'background', 'stock', 'image', 'closeup', 'jpg', 'picture', 'png', 'file', 'close up', 'pictures', 'ive', 'view', 'www', 'http', 'showing', 'blurred', 'shot', 'example', 'camera', 'footage', 'free', 'video', 'displaying', 'display', 'displayed', 'thumbnail', 'focus', 'focusing', 'detail', 'panoramic', 'standard', 'portrait', 'zoom', 'zoomed', 'show', 'showed', 'real', 'icon', 'pixelated', 'cropped', 'autofocus', 'caption', 'graphic', 'defocused', 'zoomed', ' pre ', 'available', 'royalty', 'etext', 'blurry', 'new', 'pic', 'left', 'houzz', 'full', 'small', 'br', 'looking', 'pro', 'angle', 'logo', 'close', 'right', 'blur', 'preview', 'wallpaper', 'dont', 'fixed', 'closed', 'open', 'profile', 'close', 'color', 'photo', 'colored', 'video', 'banner', 'macro', 'frame', 'cut', 'livescience', 'bottom', 'corner', 'tvmdl', 'overlay', 'original', 'sign', 'old', 'extreme', 'hq', 'isolated', 'figure', 'stockfoto', 'vrr', 'cm', 'photography', 'print', 'embedded', 'smaller', 'testing', 'captioned', 'year', 'photograph', '', 'selective', 'photoshopped', 'come', 'org', 'akc', 'iphone']


class LAION(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.caption_file = f'{root}/metadata/captions.txt'
        
    def __getitem__(self, index):
        np_array = np.load(f'{self.root}/text_emb_filtered/text_emb_{index}.npy')
        if self.transform:
            return self.transform(np_array)
        else:
            return np_array
            
    def __len__(self):
        return 410
    
    def get_captions(self, indices):
        return [linecache.getline(self.caption_file, i + 1).rstrip() for i in indices]


class FeatureGroupOutput:
    '''A model target for GradCAM that consolidates the outputs of all individual features in a group.
    See ClassifierOutputTarget in pytorch_grad_cam/utils
    '''
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.features].sum()
        return model_output[:, self.features].sum(dim = 1)


def get_grad_cam(cam, images, targets):
    images.requires_grad = True
    grayscale_cam = cam(input_tensor = images, targets = targets)
    
    return grayscale_cam


def get_clip_image_features(clip_model, probe_dataset, sample_indices, batch_size = 128, target_feature_group = [], grad_cam = None, resize_transform = None):
    
    images = torch.zeros([sample_indices.shape[0], 3, 224, 224])
    j = 0
    for i in sample_indices.long():
        images[j] = probe_dataset[i.item()][0]
        j += 1
    
    if len(target_feature_group) > 0:
        feat_grad = [FeatureGroupOutput(torch.LongTensor(target_feature_group)) for i in range(images.shape[0])]
    
    cropped_images = []
    clip_image_features = []
    i = 0
    while i * batch_size < images.shape[0]:
        image_batch = images[i * batch_size : (i + 1) * batch_size]

        if grad_cam:
            # 1. Transform with grad cam
            grayscale_mask = get_grad_cam(cam = grad_cam,
                                          images = image_batch,
                                          targets = feat_grad)

            # Crop images to only keep the activated part 
            grayscale_mask = torch.Tensor(grayscale_mask)
            grayscale_mask[grayscale_mask < 0.6] = 0
            grayscale_mask[grayscale_mask >= 0.6] = 1
            
            bb = [torch.LongTensor(mask_to_boxes(m)) for m in grayscale_mask.cpu().numpy()]
            
            image_batch = torch.cat([resize_transform(image_batch[img, : , bb[img][1]:bb[img][3] , bb[img][0]:bb[img][2]].unsqueeze(0)) for img in range(image_batch.shape[0])])

            cropped_images.append(image_batch)

        with torch.no_grad():
            clip_image_features.append(clip_model.encode_image(image_batch.cuda()).cuda())

        i += 1
        
    if len(clip_image_features) > 0:
        clip_image_features = torch.cat(clip_image_features, dim = 0)
        clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)  
    else:
        clip_image_features = torch.Tensor([]).cuda()

    cropped_images = images if len(cropped_images) == 0 else torch.cat(cropped_images)
    
    return images, clip_image_features, cropped_images
            

def mask_to_boxes(mask):
    ''' Convert a boolean (Height x Width) mask into a (N x 4) array of NON-OVERLAPPING bounding boxes
    surrounding 'islands of truth' in the mask.  Boxes indicate the (Left, Top, Right, Bottom) bounds
    of each island, with Right and Bottom being NON-INCLUSIVE (ie they point to the indices AFTER the island).

    This algorithm (Downright Boxing) does not necessarily put separate connected components into
    separate boxes.

    You can 'cut out' the island-masks with
        boxes = mask_to_boxes(mask)
        island_masks = [mask[t:b, l:r] for l, t, r, b in boxes]
    '''
    max_ix = max(s+1 for s in mask.shape)   # Use this to represent background
    # These arrays will be used to carry the 'box start' indices down and to the right.
    x_ixs = np.full(mask.shape, fill_value=max_ix)
    y_ixs = np.full(mask.shape, fill_value=max_ix)

    # Propagate the earliest x-index in each segment to the bottom-right corner of the segment
    for i in range(mask.shape[0]):
        x_fill_ix = max_ix
        for j in range(mask.shape[1]):
            above_cell_ix = x_ixs[i-1, j] if i>0 else max_ix
            still_active = mask[i, j] or ((x_fill_ix != max_ix) and (above_cell_ix != max_ix))
            x_fill_ix = min(x_fill_ix, j, above_cell_ix) if still_active else max_ix
            x_ixs[i, j] = x_fill_ix

    # Propagate the earliest y-index in each segment to the bottom-right corner of the segment
    for j in range(mask.shape[1]):
        y_fill_ix = max_ix
        for i in range(mask.shape[0]):
            left_cell_ix = y_ixs[i, j-1] if j>0 else max_ix
            still_active = mask[i, j] or ((y_fill_ix != max_ix) and (left_cell_ix != max_ix))
            y_fill_ix = min(y_fill_ix, i, left_cell_ix) if still_active else max_ix
            y_ixs[i, j] = y_fill_ix

    # Find the bottom-right corners of each segment
    new_xstops = np.diff((x_ixs != max_ix).astype(np.int32), axis=1, append=False)==-1
    new_ystops = np.diff((y_ixs != max_ix).astype(np.int32), axis=0, append=False)==-1
    corner_mask = new_xstops & new_ystops
    y_stops, x_stops = np.array(np.nonzero(corner_mask))

    # Extract the boxes, getting the top-right corners from the index arrays
    x_starts = x_ixs[y_stops, x_stops]
    y_starts = y_ixs[y_stops, x_stops]
    ltrb_boxes = np.hstack([x_starts[:, None], y_starts[:, None], x_stops[:, None]+1, y_stops[:, None]+1])
    
    max_area_box = None
    max_area = 0
    for bx in ltrb_boxes:
        ar = (bx[2] - bx[0]) * (bx[3] - bx[1])
        if ar > max_area:
            max_area = ar
            max_area_box = bx
        
    return max_area_box


def caption_images(image_emb, caption_dataloader, caption_dataset):
    top_captions_values = torch.zeros((image_emb.shape[0], 5)).half().cuda()
    top_captions_indices = torch.zeros((image_emb.shape[0], 5)).long().cuda()

    size = 0
    
    for i, (text_emb) in enumerate(caption_dataloader):
        text_emb = text_emb.view(-1, text_emb.shape[-1]).cuda()
        values, indices = (100.0 * image_emb @ text_emb.T).softmax(dim=-1).topk(5)

        cat_values = torch.cat([top_captions_values, values], dim = 1)
        cat_indices = torch.cat([top_captions_indices, indices + size], dim = 1)

        values, indices = cat_values.topk(5)
        top_captions_values = values
        top_captions_indices = torch.cat([cat_indices[i, indices[i]].unsqueeze(0) for i in range(cat_indices.shape[0])])

        size += text_emb.shape[0]
        i += 1
        
    print('SIZE:', size)

    caption_set = []
    score_set = []
    for i in range(top_captions_indices.shape[0]):
        caption_set.append(caption_dataset.get_captions(top_captions_indices[i].cpu().numpy().tolist()))
        score_set.append([val.item() for val in top_captions_values[i]])
        
    return caption_set, score_set

    
def clean_text(text):
    text = text.lower()
    text = re.sub(''', '', text)
    text = re.sub('[^A-Za-z0-9 \n']+', ' ', text)
    text = re.sub('fig\d+', ' ', text)
    text = re.sub(' . ', ' ', text)
    text = ' '.join([t for t in text.split(' ') if t not in _DISCARD_WORDS])
    return text
    
def get_max_score_for_word(words, words_dict, score):
    for n in words:
        n = n.lemmatize()
        if n not in stopwords.words('english') and n not in _DISCARD_WORDS and not any(char.isdigit() for char in n) and len(n) > 2:
            sc = score 
            words_dict[n] = sc if n not in words_dict else max(words_dict[n], sc)

    return words_dict
 

def caption_images_blip(blip_model, images):
    caption_set = []
    score_set = []
    for img in images:
        caption_set.append(blip_model.generate({'image': img.unsqueeze(0).half().cuda()}, use_nucleus_sampling=True, num_captions=5))
        score_set.append([1, 1, 1, 1, 1])
       
    return caption_set, score_set


def parse_captions(cap_set, score_set):
    all_noun_phrases = {}
    all_words = {}
    all_nouns = {}
    all_verbs = {}
    all_adj = {}
    
    for cp in range(len(cap_set)):
        caps = cap_set[cp]
        
        sim_scores = score_set[cp]
        
        noun_phrases_for_image = {}
        words_for_image = {}
        
        nouns_for_image = {}
        verbs_for_image = {}
        adj_for_image = {}
        
        for i in range(5):
            cap = clean_text(caps[i])
            # blob = TextBlob(cap).correct()
            blob = TextBlob(cap)
            
            noun_phrases_in_cap = blob.noun_phrases
            words_in_cap = set([i[0] for i in blob.tags if i[1] in ['NN', 'NNP', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']])
            nouns_in_cap = set([i[0] for i in blob.tags if i[1] in ['NN', 'NNP', 'NNS']])
            verbs_in_cap = set([i[0] for i in blob.tags if i[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']])
            adj_in_cap = set([i[0] for i in blob.tags if i[1] in ['JJ', 'JJR', 'JJS']])
            # words_in_cap = blob.words
            
            noun_phrases_for_image = get_max_score_for_word(noun_phrases_in_cap, noun_phrases_for_image, sim_scores[i])
            words_for_image = get_max_score_for_word(words_in_cap, words_for_image, sim_scores[i])
            nouns_for_image = get_max_score_for_word(nouns_in_cap, nouns_for_image, sim_scores[i])
            verbs_for_image = get_max_score_for_word(verbs_in_cap, verbs_for_image, sim_scores[i])
            adj_for_image = get_max_score_for_word(adj_in_cap, adj_for_image, sim_scores[i])
            
        # words_for_image = {k:v for (k,v) in words_for_image.items()}
        
        all_noun_phrases = {k: all_noun_phrases.get(k, 0) + noun_phrases_for_image.get(k, 0) for k in set(all_noun_phrases) | set(noun_phrases_for_image)}
        all_words = {k: all_words.get(k, 0) + words_for_image.get(k, 0) for k in set(all_words) | set(words_for_image)}
        all_nouns = {k: all_nouns.get(k, 0) + nouns_for_image.get(k, 0) for k in set(all_nouns) | set(nouns_for_image)}
        all_verbs = {k: all_verbs.get(k, 0) + verbs_for_image.get(k, 0) for k in set(all_verbs) | set(verbs_for_image)}
        all_adj = {k: all_adj.get(k, 0) + adj_for_image.get(k, 0) for k in set(all_adj) | set(adj_for_image)}
    
    all_noun_phrases = sorted({k:v/len(cap_set) for (k,v) in all_noun_phrases.items()}.items(), key = lambda k: k[1], reverse = True)
    all_words = sorted({k:v/len(cap_set) for (k,v) in all_words.items()}.items(), key = lambda k: k[1], reverse = True)
    all_nouns = sorted({k:v/len(cap_set) for (k,v) in all_nouns.items()}.items(), key = lambda k: k[1], reverse = True)
    all_verbs = sorted({k:v/len(cap_set) for (k,v) in all_verbs.items()}.items(), key = lambda k: k[1], reverse = True)
    all_adj = sorted({k:v/len(cap_set) for (k,v) in all_adj.items()}.items(), key = lambda k: k[1], reverse = True)

    return all_noun_phrases, all_words, all_nouns, all_verbs, all_adj

