import os
from gensim.scripts.glove2word2vec import glove2word2vec


def reformat():
    glove_input_file = '../GloVe/glove.6B.50d.txt'
    word2vec_output_file = '../GloVe/glove.6B.50d.word2vec.txt'
    glove2word2vec(glove_input_file, word2vec_output_file)

    glove_input_file = '../GloVe/glove.6B.100d.txt'
    word2vec_output_file = '../GloVe/glove.6B.100d.word2vec.txt'
    glove2word2vec(glove_input_file, word2vec_output_file)

    glove_input_file = '../GloVe/glove.6B.200d.txt'
    word2vec_output_file = '../GloVe/glove.6B.200d.word2vec.txt'
    glove2word2vec(glove_input_file, word2vec_output_file)

    glove_input_file = '../GloVe/glove.6B.300d.txt'
    word2vec_output_file = '../GloVe/glove.6B.300d.word2vec.txt'
    glove2word2vec(glove_input_file, word2vec_output_file)


if __name__ == '__main__':
    reformat()

    os.remove('../GloVe/glove.6B.50d.txt')
    os.remove('../GloVe/glove.6B.100d.txt')
    os.remove('../GloVe/glove.6B.200d.txt')
    os.remove('../GloVe/glove.6B.300d.txt')
