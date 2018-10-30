# Data Preparation #

Create a data directory.
```bash
mkdir -p data/datasets
```

Clone and follow the steps in the Requirements section of 
[NewsQA repository](https://github.com/Maluuba/newsqa).
Clone it with this commit id: eef43f75b298021b17a1a4812bee6fd2b546b89f

Then package the dataset.

Split the data based on their scripts:
```bash
cd newsqa
python maluuba/newsqa/split_dataset.py
cd ..
```

Now, you should have a directory maluuba/newsqa/split_data which contains train.tsv, dev.tsv and test.tsv

Copy the `prep-data/split_dataset_with_nil.py` to `newsqa/maluuba/newsqa/` and execute it to generate the nil data in `split-data-nil` directory.

We convert the csv file to json for ease of use.

```bash
cd ..
mkdir -p data/datasets/newsqa
python2.7 prep-data/data_prep_json.py newsqa/maluuba/newsqa/split_data/train.csv data/datasets/newsqa/Newsqa-train-v1.1.json
python2.7 prep-data/data_prep_json.py newsqa/maluuba/newsqa/split_data/dev.csv data/datasets/newsqa/NewsQA-dev-v1.1.json
python2.7 prep-data/data_prep_json.py newsqa/maluuba/newsqa/split_data/test.csv data/datasets/newsqa/NewsQA-test-v1.1.json
```

Do the same for nil data. Example:

```bash
mkdir -p data/datasets/newsqa/orig-nil-data-newsqa/
python2.7 prep-data/data_prep_json_for_nil.py newsqa/maluuba/newsqa/split-data-nil/nil_dev.csv data/datasets/newsqa/orig-nil-data-newsqa/nil_dev.json
```

Format the data by running the following script.

```bash
python2.7 prep-data/struct_data.py
```

Then combine the nil data with original data using `prep-data/combine_nil.py`. Example:

```bash
mkdir -p data/datasets/newsqa/newsqa-combined-orig-nil-data/
python2.7 prep-data/combine_nil.py data/datasets/newsqa/NewsQA-dev-v1.1.json.configd data/datasets/newsqa/orig-nil-data-newsqa/nil_dev.json data/datasets/newsqa/newsqa-combined-orig-nil-data/dev_nil.json
```

Create the directory for embedding file.
```bash
mkdir data/embeddings
```

Download the Glove embedding file from [here](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and unzip it in `data/embeddings`.

You can download our pretrained model from [here](https://nus.edu/2z8CDvC).
```bash
mkdir data/models
```
Keep the downloaded model in `data/models/`.

