# AlloyBERT
This research presents AlloyBERT, a transformer encoder model tailored for predicting properties like elastic modulus and yield strength of alloys based on textual inputs.
![All combined](https://github.com/cakshat/AlloyBERT/assets/77221481/17d6423f-1c49-4a01-9c3d-ae95afc4d390)

## Getting Started
Clone the repository
```
$ git clone https://github.com/cakshat/AlloyBERT.git
cd AlloyBERT
```

## Datasets
For this research, we utilized two primary datasets to explore the performance of transformer models compared to shallow machine learning models in predicting target property values with text inputs.
1. Multi Principal Elemental Alloys (MPEA) dataset: This dataset, sourced from Citrine Informatics, contains mechanical properties of several alloys. We focused on predicting the experimental Young’s modulus, and the dataset comprises 1546 entries.
2. Refractory Alloy Yield Strength (RAYS) dataset: This dataset includes experimental yield strength values for refractory alloys. With 813 entries, it provides alloy composition, testing temperature from previous literature, and data from the MPEA30–32 dataset. The dataset offers average yield strength values obtained from various processing methods.

Both the datasets can be found in the data folder as : `cd data/MPEA/MPEA.csv` and `cd data/ys_clean/ys_clean.csv`.

## How to use
1. Update the `config.yaml` file with desired parameters.
2. Run `python main.py` to train the model.
3. While pretraining make sure to set the configuration to pretrain.
4. After pretraining, update the path of pretrained model and change mode to finetune.
5. Our custom trained tokenizer which was used for training can be found in tokenizer folder and can be used if required.
