# falcon-explain


## Step 1: Download and process vocabulary. ##
FALCON requires a pre-defined vocabulary which is used for preparing explainations for neural features. In our paper we use the captions of [LAION-400M](https://laion.ai/blog/laion-400-open-dataset). In fact, this dataset provides pre-computed CLIP embeddings for all the image captions in the form of .npy files. Download all text embeddings and metadata using the following commands:
```console
$wget -r -l1 -np https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/text_emb/text_emb_{0..410}.npy -nc
$wget -r -l1 -np https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/metadata/metadata_{0..410}.parquet -nc
```
We next filter out only the unique captions. Run extract_captions.py to extract and save the unique captions and their indices by reading from the metadata parquet files. Run filter_embeddings.py to filter and store only the unique caption embeddings. Going forward, we require only :

{DATASET_PATH}/embeddings/text_emb_filtered

{DATASET_PATH}/embeddings/metadata/captions.txt

## Step 2: Identify all interpretable feature groups in your target vision model
