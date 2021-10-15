# Evaluating-CMG
Implementation of the research paper "Evaluating Commit Message Generation: To BLEU Or Not To BLEU?"

## Implementation Environment

Please install the neccessary libraries before running our codes:

- python==3.6.9
- nltk==3.4.5
- numpy==1.16.5
- scikit-learn==0.22.1

## Data & Models:

The data used for our experiments can be downloaded from the Dataset folder.
- The "human_annotations.csv" file contains the human annotated raw data of size 100.
- The MCMD dataset for various programming languages (PLs) for each of the models is the form "Model_MCMD(number).csv", where the numbers stand for the respective PLs.

| Number | PL |
| ------------- | ----------------- |
| 1 | C++ (C plus plus)|
| 2 | C# (C sharp)|
| 3 | Java (Java)|
| 4 | JS (Javascript)|
| 5 | Py (Python)|

The CMG models considered in our experiments are:
- NMT
- CommitGen
- NNGen

## Running and Experimentation

### 1. RQ1: Which factors affect CMG evaluation?

- The potential factors include Length, Word Alignment, Semantic Scoring, Case Folding, Punctuation Removal and Smoothing.
- For observing the effect of a specific factor, simply run the "Factor".py file under the RQ1 folder.

### 2. RQ2: Which metric is best suited to evaluate commit messages?

- For running the Log-MNEXT metric and obtaining its correlation with human evaluation scores, simply run the "The Log-MNEXT metric.py" file.

### 3. RQ3: How do the CMG tools perform on the new metric?

- For observing the performance of Log-MNEXT metric on a specific model for a particular PL of the MCMD dataset, simply update the ```model_MCMD(Number).csv``` part by putting the model name in place of ```model```, number 1,2,3,4 or 5 in place of ```(Number)``` in the following code section of the file "Log-MNEXT performance on the models.py" file under RQ3.
```
import csv
import json

refs=[]
preds=[]
with open('model_MCMD(Number).csv') as csvfile:
    ader = csv.reader(csvfile)
    for row in ader:
        refs.append(row[0])
        preds.append(row[1])
```
