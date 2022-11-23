import re
import math
import time
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config
import utils
from collections import Counter


class DataManager:
    """
        A class which prepares data for all the six models in this project.
        The output data can be further processed in 'class ***_DataLoader' to form batches.
    """

    def __init__(self, config: Config):
        """
            Get the configuration object which contains all the information of the model to be trained.
            Get the pretrained model and the dataset loaded.
        """
        self.train_sentence = None
        self.train_sentence: list
        self.train_sen_len = None
        self.train_label = None
        self.train_e1 = None
        self.train_e2 = None

        self.test_sentence = None
        self.test_sentence: list
        self.test_sen_len = None
        self.test_label = None
        self.test_e1 = None
        self.test_e2 = None

        self.embedding_model = None
        self.word_vec_dim = None
        self.dictionary = None
        self.dictionary: dict
        self.word_embedding = None

        self.config = config

        if config.__class__.__name__ == 'Statistics':
            self.load_training_testing_set(mode='Statistics-Training')
            print('==================================================')
            self.load_training_testing_set(mode='Statistics-Testing')
            print('==================================================')
        else:
            if config.load_emb:
                self.load_word_embedding_model()
            self.load_training_testing_set(mode='Training')
            self.load_training_testing_set(mode='Testing')
            self.build_dict()
            if config.load_emb:
                self.build_word_embedding()
            config.num_embeddings = len(self.dictionary) + 1
            print('Data loading completed.')
            print('==================================================')

            del self.embedding_model

            config.train_data_size = len(self.train_sentence)
            config.test_data_size = len(self.test_sentence)
            config.train_batches = math.ceil(config.train_data_size / config.batch_size)
            config.test_batches = math.ceil(config.test_data_size / config.batch_size)

    def get_train_sentence(self):
        """
            Returns a list of training sentences in the form of embedding indices of each word.
            If a word in the sentence is not in the dictionary, it will be replaced by the most frequently occurred word
                in the training set(index '1').
            If 'trimming_and_padding' parameter in the config file is True, all the sentences will be transformed into
                the same length(according to the parameter 'max_sen_len' in the config file).
        """
        sentence_list = []
        for (i, sentence) in enumerate(self.train_sentence):
            embedded_sentence = []
            for word in sentence:
                if word in self.dictionary.keys():
                    embedded_sentence.append(self.dictionary[word])
                else:
                    embedded_sentence.append(1)

            if self.config.trimming_and_padding is True:
                if self.train_sen_len[i] < self.config.max_sen_len:
                    for j in range(self.config.max_sen_len - self.train_sen_len[i]):
                        embedded_sentence.append(0)
                elif self.train_sen_len[i] > self.config.max_sen_len:
                    if self.train_e2[i] < self.config.max_sen_len:
                        embedded_sentence = embedded_sentence[:self.config.max_sen_len]
                    else:
                        embedded_sentence = embedded_sentence[
                                            self.train_e2[i] + 1 - self.config.max_sen_len: self.train_e2[i] + 1]
            sentence_list.append(embedded_sentence)
        return sentence_list

    def get_train_sen_len(self):
        """
            Returns a list of integer, representing the lengths of the sentences in the training set.
            This can be used to mask the padded information in LSTM, etc.
            Note that this info is only valid when 'max_sen_len' is greater than the actual maximum length of sentences.
        """
        return self.train_sen_len

    def get_train_label(self):
        """
            Returns a list of one-hot vectors for each label of the training sentences.
        """
        return self.train_label

    def get_test_sentence(self):
        """
            Returns a list of testing sentences in the form of embedding indices of each word.
            If a word in the sentence is not in the dictionary, it will be replaced by the most frequently occurred word
                in the training set(index '1').
            If 'trimming_and_padding' parameter in the config file is True, all the sentences will be transformed into
                the same length(according to the parameter 'max_sen_len' in the config file).
        """
        sentence_list = []
        for (i, sentence) in enumerate(self.test_sentence):
            embedded_sentence = []
            for word in sentence:
                if word in self.dictionary.keys():
                    embedded_sentence.append(self.dictionary[word])
                else:
                    embedded_sentence.append(1)

            if self.config.trimming_and_padding is True:
                if self.test_sen_len[i] < self.config.max_sen_len:
                    for j in range(self.config.max_sen_len - self.test_sen_len[i]):
                        embedded_sentence.append(0)
                elif self.test_sen_len[i] > self.config.max_sen_len:
                    if self.test_e2[i] < self.config.max_sen_len:
                        embedded_sentence = embedded_sentence[:self.config.max_sen_len]
                    else:
                        embedded_sentence = embedded_sentence[
                                            self.test_e2[i] + 1 - self.config.max_sen_len: self.test_e2[i] + 1]
            sentence_list.append(embedded_sentence)
        return sentence_list

    def get_test_sen_len(self):
        """
            Returns a list of integer, representing the lengths of the sentences in the testing set.
            This can be used to mask the padded information in LSTM, etc.
            Note that this info is only valid when 'max_sen_len' is greater than the actual maximum length of sentences.
        """
        return self.test_sen_len

    def get_test_label(self):
        """
            Returns a list of one-hot vectors for each label of the testing sentences.
        """
        return self.test_label

    def get_word_embedding(self):
        """
            Returns a numpy matrix, containing the embedding vectors of the most frequently occurred words
                in the training set.
        """
        return self.word_embedding

    @staticmethod
    def clean_str(text):
        """
            Normalize the input sentences.
            Referenced from https://github.com/onehaitao/CR-CNN-relation-extraction .
        """
        text = text.lower()
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"that's", "that is ", text)
        text = re.sub(r"there's", "there is ", text)
        text = re.sub(r"it's", "it is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        return text.strip()

    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        """
            Transform labels into one-hot vectors.
            Referenced from https://github.com/onehaitao/CR-CNN-relation-extraction .
        """
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def load_word_embedding_model(self):
        """
            Load pretrained word embedding model from the appointed path.
            Get the dimension of word embedding simultaneously.
        """
        start = time.time()
        vocab = {}
        with open(self.config.emb_model_path, 'r', encoding='utf-8') as f:
            embedding_info = f.readline().strip().split()
            embedding_info = list(map(int, embedding_info))
            vocab_size = embedding_info[0]
            word_vec_dim = embedding_info[1]
            for _ in tqdm(range(vocab_size), desc='Loading pretrained word embedding'):
                line = f.readline().strip().split()
                word = line[0].lower()
                vector = list(map(float, line[1:]))
                vocab[word] = vector
        end = time.time()
        print('Pretrained word embedding model has been loaded from:')
        print('   ', self.config.emb_model_path)
        print('    Number of embeddings:', vocab_size)
        print('    Dimension of word vector:', word_vec_dim)
        print('    Time consumed:', round(end - start, 2), 's')
        print('==================================================')
        time.sleep(0.01)
        self.embedding_model = vocab
        self.word_vec_dim = word_vec_dim

    def load_training_testing_set(self, mode='Testing'):
        """
            Load training set and testing set.
            Args:
                mode: choose from 'Training' and 'Testing'
            Returns:
                sentences: tokenized input sentence
                labels: one-hot vector of the relation in the sentences
                e1: the index of the first entity in the sentence
                e2: the index of the second entity in the sentence
                sen_len: the length of each sentence
        """
        start = time.time()
        data = []
        if mode == 'Training' or mode == 'Statistics-Training':
            lines = [line.strip() for line in open(self.config.train_data_path)]
        else:
            lines = [line.strip() for line in open(self.config.test_data_path)]

        max_sen_len = 0
        max_entity_dist = 0

        for idx in tqdm(range(0, len(lines), 4), desc='Loading ' + mode + ' Data'):
            id = lines[idx].split("\t")[0]
            relation = lines[idx + 1]

            sentence = lines[idx].split("\t")[1][1:-1]
            sentence = sentence.replace('<e1>', ' _e11_ ')
            sentence = sentence.replace('</e1>', ' _e12_ ')
            sentence = sentence.replace('<e2>', ' _e21_ ')
            sentence = sentence.replace('</e2>', ' _e22_ ')

            sentence = self.clean_str(sentence)
            tokens = nltk.word_tokenize(sentence)

            if self.config.remove_entity_mark is True:
                e1 = tokens.index('e12') - 2
                e2 = tokens.index('e22') - 4
                tokens.remove('e11')
                tokens.remove('e12')
                tokens.remove('e21')
                tokens.remove('e22')
            else:
                e1 = tokens.index('e12') - 1
                e2 = tokens.index('e22') - 1

            if len(tokens) > max_sen_len:
                max_sen_len = len(tokens)
            if e2 - e1 > max_entity_dist:
                max_entity_dist = e2 - e1

            data.append([id, tokens, e1, e2, len(tokens), relation])

        df = pd.DataFrame(data=data, columns=['id', 'sentence', 'e1', 'e2', 'sen_len', 'relation'])
        df['label'] = [utils.class2label[r] for r in df['relation']]

        sentences = df['sentence'].tolist()

        if mode == 'Statistics-Training' or mode == 'Statistics-Testing':
            counter = np.zeros(self.config.num_category, dtype=np.int)
            labels = df['label'].tolist()
            for label in labels:
                counter[label] += 1
            total_num = np.sum(counter)
            print(mode)
            print('Category          \tNumber\tRatio\tProportion')
            print('Other             \t', counter[0], '\t  -  \t', round(counter[0] * 100 / total_num, 2), '%')
            for i in range(1, int(self.config.num_category / 2) + 1):
                num = counter[2 * i - 1] + counter[2 * i]
                ratio = max(counter[2 * i - 1], counter[2 * i]) / min(counter[2 * i - 1], counter[2 * i])
                print(utils.label2class_no_dir[i], '\t', num, '\t', round(ratio, 1), '\t',
                      round(num * 100 / total_num, 2), '%')
            print('==================================================')

        labels = df['label']
        labels_flat = labels.values.ravel()
        labels_count = np.unique(labels_flat).shape[0]
        labels = self.dense_to_one_hot(labels_flat, labels_count)
        labels = labels.astype(np.uint8)

        e1 = df['e1'].tolist()
        e2 = df['e2'].tolist()
        sen_len = df['sen_len'].tolist()

        end = time.time()
        print(mode, 'set has been loaded. ')
        print('    Max sentence length:', max_sen_len)
        print('    Max entity distance:', max_entity_dist)
        print('    Time consumed:', round(end - start, 2), 's')
        time.sleep(0.01)

        if mode == 'Training':
            self.train_sentence = sentences
            self.train_label = labels
            self.train_e1 = e1
            self.train_e2 = e2
            self.train_sen_len = sen_len
        elif mode == 'Testing':
            self.test_sentence = sentences
            self.test_label = labels
            self.test_e1 = e1
            self.test_e2 = e2
            self.test_sen_len = sen_len

    def build_dict(self):
        """
            Build a dictionary according to the training set.
        """
        start = time.time()
        word_counter = Counter()
        for sentence in self.train_sentence:
            for word in sentence:
                word_counter[word] += 1

        ls = word_counter.most_common()

        # Reserve index 0 for padding
        self.dictionary = {w[0]: index + 1 for (index, w) in enumerate(ls)}
        end = time.time()
        print('Dictionary built in', round(end - start, 2), 's.')

    def build_word_embedding(self):
        """
            Build word embedding matrix according to
            (1) Pretrained word embedding model(self.embedding_model)
            (2) Dictionary on the training set(self.dictionary)
        """
        start = time.time()
        word_embedding = np.random.uniform(-1.0, 1.0, size=(len(self.dictionary) + 1, self.word_vec_dim))
        word_embedding[0] = np.zeros(self.word_vec_dim, dtype=float)
        for word in self.embedding_model.keys():
            if word in self.dictionary:
                word_embedding[self.dictionary[word]] = self.embedding_model[word]
        self.word_embedding = word_embedding.astype(np.float32)
        end = time.time()
        print('Word embedding built in', round(end - start, 2), 's.')
