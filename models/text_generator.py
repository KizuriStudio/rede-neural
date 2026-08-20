import numpy as np
from ..layers import Embedding, LSTM, BiLSTM, Dense, Dropout, LayerNorm, Sequential, SelfAttention
from ..activations import ReLU, GELU, Softmax, Tanh
from ..losses import CrossEntropyLoss
from ..optimizers import AdamW
from ..utils.data import TextDataset, DataLoader
from ..utils.text import generate_text, generate_interactive
from ..utils.training import Trainer


class BaseLM:
    """Base comum pra ambos os modelos."""

    def clear_cache(self):
        pass

    def _build_optimizer(self, lr):
        return AdamW(self.params, self.grads, lr=lr, weight_decay=0.01)

    def _setup_training(self, text, epochs, batch_size, seq_length, lr,
                        val_split, temperature, top_k, seed_text, gen_every):
        dataset = TextDataset(text, seq_length)
        n = int(len(dataset) * (1 - val_split))
        train_data = _Subset(dataset, list(range(n)))
        val_data = _Subset(dataset, list(range(n, len(dataset))))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        loss_fn = CrossEntropyLoss()
        optimizer = self._build_optimizer(lr)

        if seed_text is None:
            seed_text = text[:seq_length]

        def gen_fn(model):
            print(f"\n  --- Geracao (temp={temperature}, top_k={top_k}) ---")
            sample = generate_text(model, dataset, seed_text, length=200,
                                   temperature=temperature, top_k=top_k)
            print(f"  {sample}")
            print(f"  --- Fim ---")

        trainer = Trainer(self, loss_fn, optimizer, scheduler="warmup_cosine")
        return trainer, train_loader, val_loader, dataset, gen_fn

    def generate(self, dataset, seed_text, length=200, temperature=0.8, top_k=40):
        return generate_text(self, dataset, seed_text, length, temperature, top_k)

    def generate_interactive(self, dataset, seed="O", length=500, temperature=0.7, top_k=40):
        generate_interactive(self, dataset, seed, length, temperature, top_k)

    def parameters_count(self):
        total = 0
        for layer in self.layers:
            for k, v in layer.params.items():
                total += v.size
        return total


class TextGenerator(BaseLM):
    """
    Language Model baseado em LSTM para geracao de texto.

    Arquitetura:
      Embedding -> BiLSTM -> LayerNorm -> Dense -> ReLU -> Dense -> Softmax
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = []
        self.layers.append(Embedding(vocab_size, embed_dim))

        for i in range(num_layers):
            in_d = embed_dim if i == 0 else hidden_dim * 2
            self.layers.append(BiLSTM(in_d, hidden_dim))
            self.layers.append(LayerNorm(hidden_dim * 2))
            self.layers.append(Dropout(dropout))

        self.layers.append(Dense(hidden_dim * 2, hidden_dim))
        self.layers.append(ReLU())
        self.layers.append(Dropout(dropout))
        self.layers.append(Dense(hidden_dim, vocab_size))

        self._sync_params()

    def _sync_params(self):
        self.params = {}
        self.grads = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.params.items():
                key = f"{i}.{k}"
                self.params[key] = v
                self.grads[key] = np.zeros_like(v)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, grad):
        for i in reversed(range(len(self.layers))):
            grad = self.layers[i].backward(grad)
        self._copy_grads_from_layers()
        return grad

    def _copy_grads_from_layers(self):
        for i, layer in enumerate(self.layers):
            for k, v in layer.grads.items():
                key = f"{i}.{k}"
                if key in self.grads and v is not None:
                    if self.grads[key].shape == v.shape:
                        np.copyto(self.grads[key], v)
                    else:
                        self.grads[key] = v.ravel()[:self.grads[key].size].reshape(self.grads[key].shape)

    def train(self):
        for l in self.layers:
            l.train()

    def eval(self):
        for l in self.layers:
            l.eval()

    def fit(self, text, epochs=50, batch_size=32, seq_length=64, lr=0.001,
            val_split=0.1, temperature=0.8, top_k=40, seed_text=None, gen_every=5):
        trainer, train_loader, val_loader, dataset, gen_fn = self._setup_training(
            text, epochs, batch_size, seq_length, lr, val_split, temperature, top_k, seed_text, gen_every)
        trainer.fit(train_loader, val_loader, epochs=epochs, lr=lr,
                    gen_fn=gen_fn, gen_every=gen_every)
        return trainer.history


class TransformerTextGenerator(BaseLM):
    """
    Language Model baseado em Transformer.

    Arquitetura:
      Embedding -> Positional Encoding -> [SelfAttention -> FFN] x N -> Softmax
    """

    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, ffn_dim=256, dropout=0.1, max_len=512):

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.layers = []
        self.layers.append(Embedding(vocab_size, embed_dim))

        pe = self._positional_encoding(max_len, embed_dim)
        self.pos_encoding = pe

        for _ in range(num_layers):
            self.layers.append(SelfAttention(embed_dim, num_heads, dropout))
            self.layers.append(LayerNorm(embed_dim))
            self.layers.append(Dense(embed_dim, ffn_dim))
            self.layers.append(GELU())
            self.layers.append(Dense(ffn_dim, embed_dim))
            self.layers.append(LayerNorm(embed_dim))

        self.layers.append(Dense(embed_dim, vocab_size))

        self._sync_params()

    def _positional_encoding(self, max_len, d_model):
        pe = np.zeros((1, max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[0, :, 0::2] = np.sin(position * div_term)
        pe[0, :, 1::2] = np.cos(position * div_term)
        return pe

    def _sync_params(self):
        self.params = {}
        self.grads = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.params.items():
                key = f"{i}.{k}"
                self.params[key] = v
                self.grads[key] = np.zeros_like(v)

    def forward(self, x):
        B, T = x.shape
        h = self.layers[0].forward(x)
        h = h + self.pos_encoding[:, :T, :]

        for i, layer in enumerate(self.layers[1:], 1):
            if isinstance(layer, SelfAttention):
                residual = h
                h = layer.forward(h)
                h = h + residual
            elif isinstance(layer, LayerNorm) and i < len(self.layers) - 1:
                residual = h
                h = layer.forward(h)
                h = h + residual
            else:
                h = layer.forward(h)

        return h

    def backward(self, grad):
        for i in reversed(range(len(self.layers))):
            grad = self.layers[i].backward(grad)
        self._copy_grads_from_layers()
        return grad

    def _copy_grads_from_layers(self):
        for i, layer in enumerate(self.layers):
            for k, v in layer.grads.items():
                key = f"{i}.{k}"
                if key in self.grads and v is not None:
                    if self.grads[key].shape == v.shape:
                        np.copyto(self.grads[key], v)
                    else:
                        self.grads[key] = v.ravel()[:self.grads[key].size].reshape(self.grads[key].shape)

    def train(self):
        for l in self.layers:
            l.train()

    def eval(self):
        for l in self.layers:
            l.eval()

    def fit(self, text, epochs=50, batch_size=32, seq_length=64, lr=0.001,
            val_split=0.1, temperature=0.8, top_k=40, seed_text=None, gen_every=5):
        trainer, train_loader, val_loader, dataset, gen_fn = self._setup_training(
            text, epochs, batch_size, seq_length, lr, val_split, temperature, top_k, seed_text, gen_every)
        trainer.fit(train_loader, val_loader, epochs=epochs, lr=lr,
                    gen_fn=gen_fn, gen_every=gen_every)
        return trainer.history


class _Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
