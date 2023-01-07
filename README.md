# bert-mlm

## A complete training BERT model to Masked Language Modeling.


First, you need to create a conda env:

```console
conda create python=3.9 --name=mlm
```

and start the conda env:

```console
conda activate mlm
```

install transformers:

```console
conda install -c conda-forge transformers
```

To train the tokenizer run the following command:

```console
python train_bpe_tokenizer.py
```

To test your tokenizer run the following command:

```console
python test_tokenizer.py
```

To train the mlm model run the following command:

```console
python train_mlm.py
```

To test your model run the following command:

```console
python test_model.py
```