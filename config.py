import math
import time
import torch
import torch.nn as nn


class Config:
    def __init__(self, model_type=None, attention_type='multi-head', scoring_model='scaled-dot', word_vec_dim=300):
        if model_type is None:
            model_type = ['att']
        assert isinstance(model_type, list)
        for t in model_type:
            assert t in ['none', 'conv', 'lstm', 'att', 'conv&att', 'lstm&att']
        assert not ('conv' in model_type and 'conv&att' in model_type)
        assert not ('lstm' in model_type and 'lstm&att' in model_type)
        assert attention_type in ['soft', 'key-value', 'multi-head']
        assert scoring_model in ['add', 'concat', 'dot', 'scaled-dot', 'bi-linear']
        assert word_vec_dim in [50, 100, 200, 300]

        # Basic settings
        self.model_type = model_type
        self.attention_type = attention_type
        self.scoring_model = scoring_model

        # Attention settings
        self.num_blocks = 6
        self.num_heads = 8
        self.pre_norm = False
        self.activation = nn.ReLU
        self.key_value_query_vec_dim = 200

        # CNN settings
        self.conv_kernel_size = 3

        # Data settings
        self.word_vec_dim = word_vec_dim
        self.num_category = 19
        self.max_sen_len = 90
        self.directionality = False
        self.remove_entity_mark = False
        self.trimming_and_padding = True
        self.train_data_path = '../SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
        self.test_data_path = '../SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
        self.train_data_size = None
        self.test_data_size = None
        self.load_emb = True
        self.num_embeddings = None
        self.emb_model_path = '../GloVe/glove.6B.' + str(word_vec_dim) + 'd.word2vec.txt'

        # Test settings
        self.load_from = '../Model/HybridModel_20221120_212549_.pt'

        # MLP settings
        self.hidden_size = int(math.sqrt(len(model_type) * word_vec_dim * self.num_category))

        # Training settings
        self.epochs = 100
        self.batch_size = 32
        self.lr = 1e-4
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.train_batches = None
        self.test_batches = None

        # Auxiliary info
        self.start_time = time.time()
        self.training_info = None
        self.print_info()

    def get_training_info(self):
        if self.training_info is None:
            self.training_info = []
            for key in self.__dict__:
                if key == 'start_time' or key == 'training_info':
                    continue
                if self.__dict__[key] is None:
                    continue
                info = key + ' : ' + str(self.__dict__[key])
                self.training_info.append(info)
        return self.training_info

    def print_info(self):
        self.get_training_info()
        print('==================================================')
        print('Training info:')
        for info in self.training_info:
            print(info)
        print('==================================================')
