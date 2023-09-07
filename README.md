# falcon-explain

# Identifying Interpretable Subspaces of Image Representations #
This is the official PyTorch implementation of the [ICML 2023 paper](https://arxiv.org/abs/2307.10504). 

## Step 1: Download and process vocabulary ##
FALCON requires a pre-defined vocabulary which is used for preparing explainations for neural features. In our paper we use the captions of [LAION-400M](https://laion.ai/blog/laion-400-open-dataset). In fact, this dataset provides pre-computed CLIP embeddings for all the image captions in the form of .npy files. Download all text embeddings and metadata using the following commands:
```console
$wget -r -l1 -np https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/text_emb/text_emb_{0..410}.npy -nc
$wget -r -l1 -np https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/metadata/metadata_{0..410}.parquet -nc
```
We next filter out only the unique captions. Run extract_captions.py to extract and save the unique captions and their indices by reading from the metadata parquet files. Run filter_embeddings.py to filter and store only the unique caption embeddings. Going forward, we require only :

{DATASET_PATH}/embeddings/text_emb_filtered

{DATASET_PATH}/embeddings/metadata/captions.txt

## Step 2: Identify all interpretable feature groups in your target vision model
Download/install [CLIP](https://github.com/openai/CLIP) and [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam). Run get_interpretable_features.py for your target vision model using a probe image dataset of your choice. The target layer is the final pre-FC layer by default but this can be modified to be the layer of your choice. For example, we can discover the interpretable feature groups of the final representation layer of a supervised ResNet18 encoder, using ImageNet as follows:
```console
$python get_interpretable_features.py --backbone resnet18 --encoder_ckpt {PATH_TO_RESNET18_CHECKPOINT} --exp_name resnet18_imagenet --probe_dataset {PATH_TO_IMAGENET}
```

## Citation ##
Please cite our work, if you find this repository useful
```

@InProceedings{pmlr-v202-kalibhat23a,
  title = 	 {Identifying Interpretable Subspaces in Image Representations},
  author =       {Kalibhat, Neha and Bhardwaj, Shweta and Bruss, C. Bayan and Firooz, Hamed and Sanjabi, Maziar and Feizi, Soheil},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {15623--15638},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/kalibhat23a/kalibhat23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/kalibhat23a.html}
}
```
