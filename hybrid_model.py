import numpy as np
import torch
import torch.nn as nn
from config import Config
from dataloader import DataPackage
from attention_model import build_attention_model
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class HybridModel(nn.Module):
    def __init__(self, config: Config, datamanager):
        super().__init__()
        self.config = config
        if config.load_emb:
            self.word_embedding = nn.Embedding.from_pretrained(
                embeddings=torch.tensor(datamanager.get_word_embedding()),
                freeze=True,
                padding_idx=0
            )
        else:
            self.word_embedding = nn.Embedding(
                num_embeddings=config.num_embeddings,
                embedding_dim=config.word_vec_dim,
                padding_idx=0
            )

        for single_model_type in config.model_type:
            if single_model_type in ['conv', 'conv&att']:
                self.conv = nn.Conv1d(
                    in_channels=config.word_vec_dim,
                    out_channels=config.word_vec_dim,
                    kernel_size=config.conv_kernel_size,
                    bias=True,
                    padding=config.conv_kernel_size // 2
                )
                if single_model_type == 'conv&att':
                    self.conv_att = build_attention_model(config)
            elif single_model_type in ['lstm', 'lstm&att']:
                self.lstm = torch.nn.LSTM(
                    input_size=config.word_vec_dim,
                    hidden_size=config.word_vec_dim,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True
                )
                if single_model_type == 'lstm&att':
                    self.lstm_att = build_attention_model(config)
            elif single_model_type == 'att':
                self.att_layer = build_attention_model(config)

        self.dropout = nn.Dropout()
        self.activation = config.activation()
        self.max_pool = nn.MaxPool1d(kernel_size=config.max_sen_len)
        self.l1 = nn.Linear(
            in_features=config.word_vec_dim * len(config.model_type),
            out_features=config.hidden_size
        )
        self.l2 = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.num_category
        )

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=config.lr
        )

    def conv_layer(self, x):
        out = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.dropout(out)
        if 'conv&att' in self.config.model_type:
            out = self.conv_att(out)
        return out

    def lstm_layer(self, x, lengths):
        x = pack_padded_sequence(
            input=x,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )
        h, (_, _) = self.lstm(x)
        h, _ = pad_packed_sequence(
            sequence=h,
            batch_first=True,
            total_length=self.config.max_sen_len
        )
        h = h.view(-1, self.config.max_sen_len, 2, self.config.word_vec_dim)
        h = torch.sum(h, dim=2)
        out = self.activation(h)
        if 'lstm&att' in self.config.model_type:
            out = self.lstm_att(out)
        return out

    def mlp_layer(self, feature_list):
        x = []
        for feature in feature_list:
            feature = self.activation(feature)
            feature = self.max_pool(feature.permute(0, 2, 1)).squeeze()
            x.append(feature)
        x = torch.concat(x, dim=1)
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        return x

    def forward(self, x, lengths=None):
        x = self.word_embedding(x)
        out = []
        if 'conv' in self.config.model_type or 'conv&att' in self.config.model_type:
            conv_out = self.conv_layer(x)
            out.append(conv_out)
        if 'lstm' in self.config.model_type or 'lstm&att' in self.config.model_type:
            lstm_out = self.lstm_layer(x, lengths)
            out.append(lstm_out)
        if 'att' in self.config.model_type:
            att_out = self.att_layer(x)
            out.append(att_out)
        if 'none' in self.config.model_type:
            out.append(x)
        out = self.mlp_layer(out)
        return out

    def step(self, data: DataPackage):
        result = self.forward(data.sentence, data.lengths)
        loss = self.loss(result, data.label)
        loss.backward()
        return loss.data.tolist()

    def evaluate(self, data: DataPackage):
        result = self.forward(data.sentence, data.lengths)

        predicted_result = torch.argmax(result, dim=-1)
        correct_result = torch.argmax(data.label, dim=-1)

        if self.config.directionality is False:
            predicted_result = torch.mul(predicted_result, 0.5)
            correct_result = torch.mul(correct_result, 0.5)
            relation_num = np.zeros(9, dtype=np.int)
            prediction_num = np.zeros(9, dtype=np.int)
            correct_num = np.zeros(9, dtype=np.int)
        else:
            relation_num = np.zeros(18, dtype=np.int)
            prediction_num = np.zeros(18, dtype=np.int)
            correct_num = np.zeros(18, dtype=np.int)

        predicted_result = torch.add(predicted_result, -1)
        correct_result = torch.add(correct_result, -1)
        predicted_result = predicted_result.int().cpu()
        correct_result = correct_result.int().cpu()

        for i in range(len(predicted_result)):
            if correct_result[i] != -1:
                relation_num[correct_result[i]] += 1
            if predicted_result[i] != -1:
                prediction_num[predicted_result[i]] += 1
                if predicted_result[i] == correct_result[i]:
                    correct_num[predicted_result[i]] += 1
        return relation_num, prediction_num, correct_num
