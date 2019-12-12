# Language Identification

An simple implementation of language identification (LangID) system.

### Architecture

![architecture](./architecture.png)

The model is a feedforward neural network with 3 types of input features:

- char n-gram: extracting character n-grams within each word and then embedding them into vectors through look-up tables. Since it is hard to do word tokenization for some languages and we might even do not know whether a language needs word tokenization (e.g. English v.s Chinese), we just do not apply word tokenization and filter out all the n-grams that contain space. The vobulary sizes are respectively 10000, 10000, 50000 and 50000.
- unicode block: characters from different languages can fall into different unicode blocks (ranges), and we normalize the counts of all the unicode blocks of the characters to generate features. These normalized counts are then viewed as weights for weighting all the embeddings of unicode blocks. We here use the  external resourse [Unicode Character Ranges](https://www.ling.upenn.edu/courses/Spring_2003/ling538/UnicodeRanges.html) for defining each unicode block, and there are final 122 ones (see [data/unicode-blocks.csv](data/unicode-blocks.csv)).
- word feature: extracting words seperated by space and embedding them into vectors through a look-up table. Although texts from some languages can not be tokenized into words, we apply this tokenization by space anyway, and the failed ones rely on the above two features for LangID.

The detailed dimensions are shown in the bracks in the architecture figure. Adam optimizer with learning rate 1e-3 is adopted. Batch size is 256. Only word feature embeddings are applied with 50% dropout, since the word features are dominant and we would like to mask some of their neurons randomly to force the model learn better char n-gram and unicode block representations.

### Dataset and external resources

- Training and testing: [sentences from the Tatoeba website](http://downloads.tatoeba.org/exports/sentences.tar.bz2). We extract random 10000 examples for validation and testing. We keep the languages whose number of examples > 50 and 212 languages remained.
- Unicode blocks: from website [Unicode Character Ranges](https://www.ling.upenn.edu/courses/Spring_2003/ling538/UnicodeRanges.html) (see [data/unicode-blocks.csv](data/unicode-blocks.csv)).
- ISO-639-4 look-up table: from wikipedia websites (see [./ISO-639-4.csv](./ISO-639-4.csv)). This look-up table is used to transfer each ISO-639-4 language code to its corresponding label.

### Performance



### Dependency

- python=3.6
- pandas=0.25.3
- pytorch=1.3.1
- tqdm=4.40.0
- scikit-leran=0.21.3

See also [./requirements.txt](./requirements.txt).

### Project Structure

### Usage

**Train**: 

```bash
python train.py -gpu -1 -ftrain data/train.csv -fvalid data/valid.csv -ftest data/test.csv -mdir mdl/ffd -nepoches 4
```

Model files and validation results will be saved in the directory indicated by -midr (mdl/ffd in this cae). Built feature extractors and extracted features will be saved to [./cache](./cache) directory.  The -gpu option indicates the gpu index for training, and should be set to -1 for using cpu. 

**Evaluation**:

```shell
python -ftest data/test.csv -mdir mdl/ffd -gpu -1 -bsz 256
```

**Predict from file input**:

```shell
python -finput data/input.txt -mdir mdl/ffd -gpu -1 -bsz 256
```

Each line in the input file indicated by -finput contains an input example.

**Interact with command line**:

```shell
python interactive.py -gpu -1
```











