# Details to use NAMANDA #

Create a symbolic link of the `../data` for ease of use.

```bash
ln -s ../data/ ./
```

### Format the data for preprocessing ###

Use `scripts/prepend_nil.py` to add a NIL token at the begining of every passage for easy computation of probabilities in model output. Example:

```bash
python scripts/prepend_nil.py data/datasets/newsqa/newsqa-combined-orig-nil-data/dev_nil.json
```

### Preprocessing ###

Use `scripts/reader/preprocess.py` to tokenize the data. To view the options to run:

```bash
python scripts/reader/preprocess.py -h
```

### Training ###

For trianing options execute:

```bash 
python scripts/reader/train.py -h
```

### Testing ###

See options by executing:
```bash
python scripts/reader/predict.py -h
```

For evaluation, use:
```bash
python eval_scores.py <predicted answer dictionary>
```

You can also download our [trained model](https://nus.edu/2z8CDvC) and test.

