import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class ScoringModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.max_sen_len = config.max_sen_len
        self.vec_dim = config.key_value_query_vec_dim
        self.activation = nn.Tanh()
        self.sqrt_dk = math.sqrt(self.vec_dim)

    def forward(self, h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AdditiveScoringModel(ScoringModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.W = nn.Linear(
            in_features=self.vec_dim,
            out_features=self.vec_dim,
            bias=False
        )
        self.U = nn.Linear(
            in_features=self.vec_dim,
            out_features=self.vec_dim,
            bias=False
        )
        self.v = nn.Linear(
            in_features=self.vec_dim,
            out_features=self.max_sen_len,
            bias=False
        )

    def forward(self, h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        Wh = self.W(h)
        Uq = self.U(q)
        s = self.v(self.activation(Wh + Uq))
        alpha = F.softmax(s, dim=1)
        return alpha


class ConcatenateScoringModel(ScoringModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.W = nn.Linear(
            in_features=2 * self.vec_dim,
            out_features=self.vec_dim,
            bias=False
        )
        self.v = nn.Linear(
            in_features=self.vec_dim,
            out_features=self.max_sen_len,
            bias=False
        )

    def forward(self, h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        hq = torch.concat([h, q], dim=2)
        Whq = self.W(hq)
        s = self.v(self.activation(Whq))
        alpha = F.softmax(s, dim=1)
        return alpha


class DotProductScoringModel(ScoringModel):
    def __init__(self, config: Config):
        super().__init__(config)

    def forward(self, h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        s = torch.matmul(h, q.permute(0, 2, 1))
        alpha = F.softmax(s, dim=1)
        return alpha


class ScaledDotProductScoringModel(ScoringModel):
    def __init__(self, config: Config):
        super().__init__(config)

    def forward(self, h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        s = torch.matmul(h, q.permute(0, 2, 1))
        s = s / self.sqrt_dk
        alpha = F.softmax(s, dim=1)
        return alpha


class BiLinearScoringModel(ScoringModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.W = nn.Linear(
            in_features=self.vec_dim,
            out_features=self.vec_dim,
            bias=False
        )

    def forward(self, h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        Wq = self.W(q)
        s = torch.matmul(h, Wq.permute(0, 2, 1))
        alpha = F.softmax(s, dim=1)
        return alpha


def build_scoring_model(config: Config) -> ScoringModel:
    scoring_model = config.scoring_model
    if scoring_model == 'add':
        return AdditiveScoringModel(config)
    elif scoring_model == 'concat':
        return ConcatenateScoringModel(config)
    elif scoring_model == 'dot':
        return DotProductScoringModel(config)
    elif scoring_model == 'scaled-dot':
        return ScaledDotProductScoringModel(config)
    elif scoring_model == 'bi-linear':
        return BiLinearScoringModel(config)
    else:
        info = 'The designated scoring model \'' + scoring_model + \
               '\' does not exist.\nPlease choose from [\'add\', \'concat\', \'dot\', \'scaled-dot\', \'bi-linear\'].'
        raise Exception(info)
