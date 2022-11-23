import time
import torch
import datetime
import numpy as np
from config import Config
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self, config: Config, network, dataloader):
        self.start_time = time.time()
        self.end_time = None
        self.network = network.to(config.device)
        self.dataloader = dataloader
        self.config = config
        self.best_model = None
        self.best_acc = 0.0
        self.training_log = []
        self.train_macro = []
        self.train_micro = []
        self.test_macro = []
        self.test_micro = []
        self.loss = []

    def train(self):
        self.network.train()
        print('Epoc\tTrMa\tTrMi\tTeMa\tTeMi\tLoss')

        for i in range(self.config.epochs):
            loss = 0
            self.dataloader.begin_new_epoch()
            for j in range(self.config.train_batches):
                data = self.dataloader.get_training_data()
                loss += self.network.step(data)
                self.network.optimizer.step()
                self.network.optimizer.zero_grad()
                print('\rEpoch', i + 1, 'completed:', round(j * 100 / self.config.train_batches), '%', end='')

            print('\r{:<8d}'.format(i + 1), end='')
            self.loss.append(loss)

            self.test(mode='Train')
            self.test(mode='Test')
            print(round(loss, 1))
        self.end_time = time.time()
        print('==================================================')
        print('Time consumed:', round(self.end_time - self.start_time, 1), 's')

    def test(self, mode='Test'):
        self.network.eval()

        if self.config.directionality is False:
            relation_num = np.zeros(9, dtype=np.int)
            predicted_num = np.zeros(9, dtype=np.int)
            correct_num = np.zeros(9, dtype=np.int)
        else:
            relation_num = np.zeros(18, dtype=np.int)
            predicted_num = np.zeros(18, dtype=np.int)
            correct_num = np.zeros(18, dtype=np.int)

        if mode == 'Train':
            batch_num = self.config.train_batches
        else:
            batch_num = self.config.test_batches

        for i in range(batch_num):
            if mode == 'Train':
                data = self.dataloader.get_training_data(i)
            else:
                data = self.dataloader.get_testing_data(i)
            r, p, c = self.network.evaluate(data)
            relation_num += r
            predicted_num += p
            correct_num += c
        try:
            macro_precision = correct_num / (predicted_num + 1e-3)
            macro_recall = correct_num / relation_num
            macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-3)
            macro_f1 = np.mean(macro_f1)

            micro_precision = np.sum(correct_num) / np.sum(predicted_num + 1e-3)
            micro_recall = np.sum(correct_num) / np.sum(relation_num)
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-3)

            print('{:<8.1f}{:<8.1f}'.format(macro_f1 * 100, micro_f1 * 100), end='')

            if mode == 'Train':
                self.train_macro.append(macro_f1)
                self.train_micro.append(micro_f1)
            else:
                self.test_macro.append(macro_f1)
                self.test_micro.append(micro_f1)
                if macro_f1 > self.best_acc:
                    self.best_acc = macro_f1
                    self.best_model = self.network.state_dict()
        except ZeroDivisionError:
            print('NoPre', end='\t')

        self.network.train()

    def save_model(self):
        model = self.network.__class__.__name__
        t = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = model + '_' + t + '_.pt'
        path = '../Model/' + filename
        torch.save(self.best_model, path)

    def save_training_log(self):
        model = self.network.__class__.__name__
        t = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = model + '_' + t + '_.txt'
        path = '../TrainingLog/' + filename
        file = open(path, 'w')
        training_info = self.config.get_training_info()
        for info in training_info:
            file.write(info + '\n')
        file.write('==============================\n')
        file.write('Epoc    TrMa    TrMi    TeMa    TeMi\n')
        for i in range(self.config.epochs):
            line = '{:<8d}'.format(i + 1) + '{:<8.1f}'.format(self.train_macro[i] * 100) + \
                   '{:<8.1f}'.format(self.train_micro[i] * 100) + '{:<8.1f}'.format(self.test_macro[i] * 100) + \
                   '{:<8.1f}'.format(self.test_micro[i] * 100) + '\n'
            file.write(line)
        file.write('==============================\n')
        best_macro = np.max(self.test_macro)
        best_micro = np.max(self.test_micro)
        line = 'Best macro-averaged F1: ' + str(round(best_macro * 100, 1)) + '%\n'
        file.write(line)
        line = 'Best micro-averaged F1: ' + str(round(best_micro * 100, 1)) + '%\n'
        file.write(line)
        line = 'Time consumed: ' + str(round(self.end_time - self.start_time, 1)) + 's\n'
        file.write(line)
        file.close()

    def visualize(self):
        model = self.network.__class__.__name__
        t = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = model + '_' + t + '_.jpg'
        path = '../Figure/' + filename

        x = np.array(range(1, self.config.epochs + 1))
        y1 = np.array(self.train_macro)
        y2 = np.array(self.test_macro)
        plt.figure()
        plt.xlabel('Epoch', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.ylabel('macro-F1', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.plot(x, y1, 'dodgerblue', linewidth=1.0, label='Train-F1')
        plt.plot(x, y2, 'red', linewidth=1.0, label='Test-F1')
        plt.xticks(fontproperties='Times New Roman', size=10)
        plt.yticks(fontproperties='Times New Roman', size=10)
        plt.legend(prop={'family': 'Times New Roman', 'size': 10})
        plt.savefig(path)
        plt.show()
