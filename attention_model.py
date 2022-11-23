import torch
import torch.nn as nn
from config import Config
from scoring_model import build_scoring_model


class AttentionModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.pre_norm = config.pre_norm
        self.batch_size = config.batch_size
        self.word_vec_dim = config.word_vec_dim
        self.key_value_query_vec_dim = config.key_value_query_vec_dim
        self.scoring_model = build_scoring_model(config)
        self.ffn = nn.Sequential(
            nn.Linear(
                in_features=self.word_vec_dim,
                out_features=self.word_vec_dim
            ),
            config.activation(),
            nn.Linear(
                in_features=self.word_vec_dim,
                out_features=self.word_vec_dim
            )
        )
        self.norm = nn.LayerNorm([config.max_sen_len, config.word_vec_dim])

    def att_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            norm_x = self.norm(x)
            att = self.att_forward(norm_x) + x
        else:
            att = self.att_forward(x)
            att = self.norm(att + x)
        return att


class SoftAttentionModel(AttentionModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.Wh = nn.Linear(
            in_features=self.word_vec_dim,
            out_features=self.key_value_query_vec_dim,
            bias=False
        )
        self.Wq = nn.Linear(
            in_features=self.word_vec_dim,
            out_features=self.key_value_query_vec_dim,
            bias=False
        )
        self.out_linear = nn.Linear(
            in_features=self.key_value_query_vec_dim,
            out_features=self.word_vec_dim
        )

    def att_forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.Wh(x)
        q = self.Wq(x)
        alpha = self.scoring_model(h, q)
        att = torch.matmul(alpha.permute(0, 2, 1), h)
        att = self.out_linear(att)
        return att


class KeyValueAttentionModel(AttentionModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.Wk = nn.Linear(
            in_features=self.word_vec_dim,
            out_features=self.key_value_query_vec_dim,
            bias=False
        )
        self.Wv = nn.Linear(
            in_features=self.word_vec_dim,
            out_features=self.key_value_query_vec_dim,
            bias=False
        )
        self.Wq = nn.Linear(
            in_features=self.word_vec_dim,
            out_features=self.key_value_query_vec_dim,
            bias=False
        )
        self.out_linear = nn.Linear(
            in_features=self.key_value_query_vec_dim,
            out_features=self.word_vec_dim
        )

    def att_forward(self, x: torch.Tensor, out_linear=True) -> torch.Tensor:
        k = self.Wk(x)
        v = self.Wv(x)
        q = self.Wq(x)
        alpha = self.scoring_model(k, q)
        att = torch.matmul(alpha.permute(0, 2, 1), v)
        if out_linear:
            att = self.out_linear(att)
        return att


class MultiHeadAttentionModel(AttentionModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.head_list = nn.ModuleList(KeyValueAttentionModel(config) for _ in range(config.num_heads))
        self.out_linear = nn.Linear(
            in_features=config.num_heads * self.key_value_query_vec_dim,
            out_features=self.word_vec_dim
        )

    def att_forward(self, x: torch.Tensor) -> torch.Tensor:
        att_list = [head.att_forward(x, out_linear=False) for head in self.head_list]
        att_list = torch.concat(att_list, dim=2)
        att = self.out_linear(att_list)
        return att


def build_attention_model(config: Config):
    attention_type = config.attention_type
    if attention_type == 'soft':
        model = SoftAttentionModel
    elif attention_type == 'key-value':
        model = KeyValueAttentionModel
    elif attention_type == 'multi-head':
        model = MultiHeadAttentionModel
    else:
        info = 'The designated attention type \'' + attention_type + \
               '\' does not exist.\nPlease choose from [\'soft\', \'key-value\', \'multi-head\'].'
        raise Exception(info)
    return nn.Sequential(*[model(config) for _ in range(config.num_blocks)])
