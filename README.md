# Exploration on the Implementation Details of the Attention Mechanism
### Ziao Ma, Beijing Institute of Technology
## Summary
Based on the current research results of the attention mechanism in deep learning, 
this work analyzes and explains the differences between the attention mechanism and CNN and RNN in detail, 
and explores the implementation details of the attention mechanism and the mathematical principles behind it.

## Requirements
* OS：Windows、Linux、MacOS
* Python 3.8
* PyTorch 1.13
* Numpy 1.21.5
* Pandas 1.4.2
* nltk 3.6.7
* matplotlib 3.5.1
* wget 3.2
* tqdm 4.64.0

## Explanation of data
* The GloVe 6B word embedding representation dataset used in this experiment was obtained from https://nlp.stanford.edu/projects/glove/glove.6B.zip
* The SemEval-2010 Task 8 dataset used in this experiment was obtained from http://docs.google.com/View?id=dfvxd49s_36c28v9pmw
* After obtaining the above data, you need to organize the folders into the following structure:
```plain
Parent Folder
├── code
├── SemEval2010_task8_all_data
├── GloVe
│   ├──glove.6B.50d.word2vec.txt
│   ├──glove.6B.100d.word2vec.txt
│   ├──glove.6B.200d.word2vec.txt
│   ├──glove.6B.300d.word2vec.txt
├── Model
├── TrainingLog
├── Figure
```
Due to the large size of the glove.6B.zip file, 
the connection may be interrupted when using setup.py to build the environment. 
You can download the compressed package separately. 
After decompressing, put the files in the folder into the GloVe directory, 
and then run reformat.py to prepare the data.

## Usage
* Data preparation and folder structure construction
  ```commandline
  python setup.py
  ```
* Statistics before training
  ```commandline
  python statistics.py
  ```
* Model training
  ```commandline
  python train.py
  ```

## Results
**Comparison among different model architectures**

| Model          | Macro-F1 | Micro-F1 |
|----------------|----------|----------|
| None           | 46.8     | 50.9     |
| CNN            | 78.4     | 79.8     |
| LSTM           | 74.7     | 76.7     |
| Attention      | 60.5     | 62.9     |
| CNN+Attention  | 78.6     | 80.1     |
| LSTM+Attention | 77.0     | 78.1     |

**Impact of number of layers**

| Number of layers | Macro-F1 | Micro-F1 |
|------------------|----------|----------|
| 1                | 60.5     | 62.9     |
| 2                | 60.2     | 62.7     |
| 3                | 60.6     | 63.6     |
| 4                | 60.9     | 63.7     |
| 5                | 61.6     | 63.7     |
| 6                | 62.6     | 64.4     |

**Impact of scoring models**

| Scoring model | Macro-F1 | Micro-F1 |
|---------------|----------|----------|
| Additive      | 60.4     | 62.4     |
| Concatenate   | 59.9     | 62.4     |    
| Dot Product   | 57.3     | 61.5     |   
| Scaled-Dot    | 60.5     | 62.9     |   
| Bi-linear     | 58.7     | 61.0     | 

**Impact of attention category**

| Attention Category | Macro-F1 | Micro-F1 |
|--------------------|----------|----------|
| Soft               | 60.3     | 62.5     |
| Key-Value          | 60.5     | 62.9     |
| Multi-head         | 61.2     | 63.3     |

**Impact of norm position**

| Norm Position | Macro-F1 | Micro-F1 |
|---------------|----------|----------|
| Pre-norm      | 56.2     | 58.8     |
| Post-norm     | 60.5     | 62.9     |

**Overall experiment**

| Model settings   | Macro-F1 | Micro-F1 |
|------------------|----------|----------|
| Soft+Dot Product | 60.1     | 62.1     |
| +Scaled-Dot      | 60.3     | 62.5     |
| +Key-Value       | 60.5     | 62.9     |
| +Multi-layer     | 62.6     | 64.4     |
| +Multi-head      | 64.9     | 66.3     |

**Note**: Since the random number generator seed was not used in this experiment, the results of repeated experiments may have a deviation of no more than 2%.
The model codes in this project are all implemented by myself. 
A small amount of code in the data preprocessing part is a direct reference to other people's code, which is also clearly marked in the function documentation and a link to the source code is provided.
