# cera-summ
Source code for the paper: **"Supervising the Centroid Baseline for Extractive Multi-Document Summarization"** <br>
Authors: Simão Gonçalves, Gonçalo Correia, Diogo Pernes, Afonso Mendes<br>
Presented at: "The 4th New Frontiers in Summarization (with LLMs) Workshop (NewSumm Workshop 2023)"<br>


The code in this project provides the summarization algorithm described in the paper:<br>

  - Multilingual sentence embeddings allowing **cross-lingual extractive summarization**;<br>
  - **Beam Search** and **Greedy Search** algorithms to facilitate a thorough exploration of the candidate space, ultimately enhancing summary quality;<br>
  - **Centroid Regression Attention models** that approximate the oracle centroid obtained from the ground-truth target summary.<br>

## Setup
Create a virtual environment and install the required packages

```
conda create -n cera-summ python=3.10.10
pip install -r requirements.txt
```

Download the [TAC2008](https://tac.nist.gov/2008/summarization/) and [DUC2004](https://github.com/UsmanNiazi/DUC-2004-Dataset) from the official providers.
To download our pre-processed version of the CrossSum dataset, run:
```
curl ftp://ftp.priberam.com/cera-summ/crosssum_data.tar.gz --user "ftp.priberam.com|anonymous":anonymous -o ./crosssum_data.tar.gz
tar -xzvf crosssum_data.tar.gz
```
By using this data you are agreeing with the license terms of the [original dataset](https://github.com/csebuetnlp/CrossSum).

To download the trained models run:
```
curl ftp://ftp.priberam.com/cera-summ/cera-summ_models.tar.gz --user "ftp.priberam.com|anonymous":anonymous -o ./cera-summ_models.tar.gz
tar -xzvf cera-summ_models.tar.gz
```


## Model training

E.g. To train the CeRAI model on the CrossSum dataset, run:
```
python CLI.py fit -c centroid_attention/cross_sum_config.yaml --model.interpolation 1 --data.train_dataset_path ./CrossSum/train.jsonl --data.validation_dataset_path ./CrossSum/val.jsonl --data.test_dataset_path ./CrossSum/test.jsonl --R2_R_checkpoint.filename ./checkpoints/CeRAI-CrossSum-BestR2R.ckpt
```

**Further configuration options**:<br>
  - On the folder "centroid_attention", edit the ".yaml" file corresponding to the dataset in which you want to train your model;
  - The default hyperparameters in the "trainer->model" field were the ones we used for training our models;
  - **If you don't have the data stored in disk (1st run)** On the "trainer->data->init_args" field set "use_pickle" to ```false``` if it is the first time you are training the model on the
     selected dataset. Also, if you want to (we strongly suggest), you can set "save_pickle" to ```true``` to save the data generated so that next time you train the model you can
     use the data saved in disk instead of waiting for it to be generated again;
 - **If you have the data stored in disk** On the "trainer->data->init_args" field set "use_pickle" to ```true``` and point "pickle_path" to the saved pickle file that was generated on
     previous training runs. The data loading process should be faster;
 -  To train a **CeRA** model, "trainer->model->interpolation" should be set to ```false```. To train a **CeRAI** model, set "trainer->model->interpolation" to ```true```;
 -  On "trainer->logger" select the "class_path" logger you want to use;
 -  On "trainer->logger->init_args" choose a name for your run ("run_name"), the experiment name ("experiment_name"), and a "tracking_uri" with an url for the logger page
     where your experiment will run;
 -  Under "trainer", on the "default_root_dir", select a path to a folder where your model checkpoints will be saved. Note that by default we are storing two checkpoints per model,
     one on the best validation loss and another one on the best R2-R score found on the validation set (these configs can be changed under "trainer->val_loss_checkpoint" and
     "trainer->R2_R_checkpoint");
 -  By default we are using early stopping on the R2-R score reported during validation, this can also be changed under "trainer->early_stopping".


## Model Evaluation

E.g. To evaluate the CeRAI model and the summarization algorithm on the CrossSum-ZS dataset, run:
```
python main.py --dataset_path ./CrossSum/test_zs.jsonl --centroid_model_path ./checkpoints/CeRAI-CrossSum-BestR2R.ckpt --partition test --dataset_name CrossSum --reference_type single_ref --summarizer_type ghalandari --sentences n_first --n 9 --budget 100 --beam_width 5 --counter_limit 9 --alpha 0 --centroid_type estimated
```

## Citation
TODO


