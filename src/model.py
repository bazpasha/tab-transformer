import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, normal_


class Attention(nn.Module):
    def __init__(self, d_model, attention_function=F.softmax, n_fixed_queries=None):
        super().__init__()

        self.d_model = d_model
        self.attention_function = attention_function
        self.keys = nn.Linear(d_model, d_model, bias=False)
        self.values = nn.Linear(d_model, d_model, bias=False)

        if n_fixed_queries is not None:
            self.queries = nn.Parameter(data=torch.empty(1, n_fixed_queries, d_model))
        else:
            self.queries = nn.Linear(d_model, d_model, bias=False)
        self.fixed_queries = n_fixed_queries is not None

        self._initialize()

    def forward(self, x, mask=None, attn_weights=False):
        # x.shape = [batch, length, token_dim]
        keys = self.keys(x)
        values = self.values(x)
        if self.fixed_queries:
            queries = self.queries
        else:
            queries = self.queries(x)

        # dot_products.shape = [batch, length (query), length (key)]
        dot_products = torch.matmul(queries, keys.transpose(1, 2))
        if mask is not None:
            dot_products = dot_products * mask - 1e9 * (1 - mask)

        weights = self.attention_function(dot_products / np.sqrt(self.d_model), dim=-1)
        result = torch.matmul(weights, values)

        if attn_weights:
            return result, weights
        return result

    def _initialize(self):
        for linear in [self.keys, self.values]:
            normal_(linear.weight, std=0.1)
        if self.fixed_queries:
            normal_(self.queries, std=0.1)
        else:
            normal_(self.queries.weight, std=0.1)


class SparseTransformerEncoder(nn.Module):
    def __init__(self, d_model, dim_feedforward, attention_kwargs=None):
        super().__init__()

        attention_kwargs = attention_kwargs or {}
        self.input_norm = nn.LayerNorm(d_model)
        self.self_attention = Attention(d_model, **attention_kwargs)
        self.middle_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

        self._initialize()

    def forward(self, x):
        # x.shape = [batch, length, token_dim]
        x = self.input_norm(x)
        x = self.middle_norm(self.self_attention(x) + x)
        return self.ff(x) + x

    def _initialize(self):
        for layer in self.ff:
            if type(layer) == nn.Linear:
                normal_(layer.weight, std=0.1)
                constant_(layer.bias, val=0)


class TabTransformer(nn.Module):
    def __init__(
        self,
        n_features,
        n_tokens,
        d_model=64,
        n_transformers=4,
        dim_feedforward=128,
        dim_output=1,
        attention_kwargs=None,
        transformer_kwargs=None,
    ):
        super().__init__()

        attention_kwargs = attention_kwargs or {}
        transformer_kwargs = transformer_kwargs or {}

        self.n_tokens = n_tokens
        self.linear_embeddings = nn.Parameter(data=torch.empty(1, n_features, d_model))
        self.const_embeddings = nn.Parameter(data=torch.empty(1, n_features, d_model))

        self.tokenizer = Attention(d_model, n_fixed_queries=n_tokens, **attention_kwargs)
        transformer_list = [
            SparseTransformerEncoder(d_model, dim_feedforward, attention_kwargs, **transformer_kwargs)
            for _ in range(n_transformers)
        ]
        self.transformer = nn.Sequential(*transformer_list)

        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, dim_output)
        self._initialize()

    def forward(self, x):
        n = x.size(0)
        x = x.unsqueeze(-1) * self.linear_embeddings + self.const_embeddings.expand(n, -1, -1)
        x = self.tokenizer(x)
        x = self.transformer(x)
        x = self.norm(x).mean(dim=1)
        return self.output(x).squeeze(1)

    def _initialize(self):
        normal_(self.linear_embeddings, std=0.1)
        normal_(self.const_embeddings, std=0.1)
        normal_(self.output.weight, std=0.1)
        constant_(self.output.bias, val=0)
