import torch
import random
from config import Config


class Dataloader:
    def __init__(self, config: Config, datamanager):
        self.config = config

        self.train_sentence = datamanager.get_train_sentence()
        self.train_label = datamanager.get_train_label()
        self.train_lengths = datamanager.get_train_sen_len()

        self.test_sentence = datamanager.get_test_sentence()
        self.test_label = datamanager.get_test_label()
        self.test_lengths = datamanager.get_test_sen_len()

        self.batch_no = None
        self.train_index = None

    def begin_new_epoch(self):
        x = [i for i in range(len(self.train_sentence))]
        random.shuffle(x)
        self.train_index = x
        self.batch_no = 0

    def get_training_data(self, batch_no=None):
        if batch_no is None:
            if self.batch_no != self.config.train_batches - 1:
                index_list = self.train_index[
                             self.batch_no * self.config.batch_size: (self.batch_no + 1) * self.config.batch_size]
            else:
                index_list = self.train_index[self.batch_no * self.config.batch_size:]
            self.batch_no += 1
        else:
            if batch_no != self.config.train_batches - 1:
                index_list = range(batch_no * self.config.batch_size, (batch_no + 1) * self.config.batch_size)
            else:
                index_list = range(batch_no * self.config.batch_size, len(self.train_sentence))

        sentence_list = []
        label_list = []
        length_list = []

        for i in index_list:
            sentence = torch.tensor(self.train_sentence[i], dtype=torch.int)
            sentence = torch.reshape(sentence, [1, -1])
            sentence_list.append(sentence)

            label = torch.tensor(self.train_label[i], dtype=torch.float32)
            label = torch.reshape(label, [1, -1])
            label_list.append(label)

            length_list.append(self.train_lengths[i])

        sentence_batch = torch.cat(sentence_list, dim=0)
        label_batch = torch.cat(label_list, dim=0)
        lengths = torch.tensor(length_list)

        return DataPackage(
            sentence=sentence_batch,
            label=label_batch,
            lengths=lengths,
            config=self.config
        )

    def get_testing_data(self, batch_no):
        if batch_no != self.config.test_batches - 1:
            index_list = range(batch_no * self.config.batch_size, (batch_no + 1) * self.config.batch_size)
        else:
            index_list = range(batch_no * self.config.batch_size, len(self.test_sentence))

        sentence_list = []
        label_list = []
        length_list = []

        for i in index_list:
            sentence = torch.tensor(self.test_sentence[i], dtype=torch.int)
            sentence = torch.reshape(sentence, [1, -1])
            sentence_list.append(sentence)

            label = torch.tensor(self.test_label[i], dtype=torch.float32)
            label = torch.reshape(label, [1, -1])
            label_list.append(label)

            length_list.append(self.test_lengths[i])

        if len(label_list) < self.config.batch_size:
            padding_sentence = torch.zeros(self.config.max_sen_len, dtype=torch.int)
            padding_sentence = torch.reshape(padding_sentence, [1, -1])
            padding_label = torch.zeros(self.config.num_category, dtype=torch.float32)
            padding_label = torch.reshape(padding_label, [1, -1])
            for i in range(self.config.batch_size - len(label_list)):
                sentence_list.append(padding_sentence)
                label_list.append(padding_label)
                length_list.append(1)

        sentence_batch = torch.cat(sentence_list, dim=0)
        label_batch = torch.cat(label_list, dim=0)
        lengths = torch.tensor(length_list)

        return DataPackage(
            sentence=sentence_batch,
            label=label_batch,
            lengths=lengths,
            config=self.config
        )


class DataPackage:
    def __init__(self, sentence, label, lengths, config):
        self.sentence = sentence.to(config.device)
        self.label = label.to(config.device)
        self.lengths = lengths
