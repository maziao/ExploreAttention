import os
import wget
import zipfile
from reformat import reformat


if __name__ == '__main__':
    folders = ['../Model', '../Figure', '../TrainingLog']
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)

    if not os.path.exists('../GloVe/glove.6B.50d.word2vec.txt'):
        wget.download('https://nlp.stanford.edu/data/glove.6B.zip', '../glove.6B.zip')
        with zipfile.ZipFile('../glove.6B.zip') as zf:
            zf.extractall()
        os.rename('../glove.6B', '../GloVe')
        os.remove('../glove.6B.zip')

        reformat()

        os.remove('../GloVe/glove.6B.50d.txt')
        os.remove('../GloVe/glove.6B.100d.txt')
        os.remove('../GloVe/glove.6B.200d.txt')
        os.remove('../GloVe/glove.6B.300d.txt')
