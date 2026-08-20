import numpy as np
from ..layers import Embedding, Dense, Dropout, LayerNorm
from ..layers.rmsnorm import RMSNorm
from ..layers.swiglu import SwiGLU
from ..layers.causal_attention import CausalSelfAttention
from ..losses import CrossEntropyLoss
from ..optimizers import AdamW
from ..utils.data import TextDataset, DataLoader
from ..utils.sampling import generate_advanced, generate_stream
from ..utils.training import Trainer


class GPTBlock:
    """Bloco transformer estilo GPT-2/LLaMA com Pre-Norm.
    Pre-Norm: RMSNorm → Attention → + Residual → RMSNorm → SwiGLU → + Residual"""

    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1, max_len=512, use_rope=True, use_rmsnorm=True):
        self.use_rmsnorm = use_rmsnorm

        if use_rmsnorm:
            self.norm1 = RMSNorm(embed_dim)
            self.norm2 = RMSNorm(embed_dim)
        else:
            self.norm1 = LayerNorm(embed_dim)
            self.norm2 = LayerNorm(embed_dim)

        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout, max_len, use_rope=use_rope)
        self.ffn = SwiGLU(embed_dim, ffn_dim)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x, use_cache=False, pos_offset=0):
        h = self.norm1.forward(x)
        h = self.attn.forward(h, use_cache=use_cache, pos_offset=pos_offset)
        x = x + self.dropout1.forward(h)

        h = self.norm2.forward(x)
        h = self.ffn.forward(h)
        x = x + self.dropout2.forward(h)

        return x

    def backward(self, grad):
        h = self.dropout2.backward(grad)
        h = self.ffn.backward(h)
        h = self.norm2.backward(h)
        grad = grad - h
        h = self.dropout1.backward(grad)
        h = self.attn.backward(h)
        h = self.norm1.backward(h)
        return grad - h

    def train(self):
        for m in [self.norm1, self.norm2, self.attn, self.ffn, self.dropout1, self.dropout2]:
            m.train()

    def eval(self):
        for m in [self.norm1, self.norm2, self.attn, self.ffn, self.dropout1, self.dropout2]:
            m.eval()

    def clear_cache(self):
        self.attn.clear_cache()

    @property
    def params(self):
        p = {}
        for prefix, m in [("n1.", self.norm1), ("n2.", self.norm2),
                          ("attn.", self.attn), ("ffn.", self.ffn)]:
            for k, v in m.params.items():
                p[prefix + k] = v
        return p

    @property
    def grads(self):
        g = {}
        for prefix, m in [("n1.", self.norm1), ("n2.", self.norm2),
                          ("attn.", self.attn), ("ffn.", self.ffn)]:
            for k, v in m.grads.items():
                g[prefix + k] = v
        return g


class GPTDecoder:
    """
    GPT-style Decoder-Only Transformer.

    Arquitetura estilo GPT-2/LLaMA:
      Token Embedding + RoPE
      ↓
      [RMSNorm → CausalAttention → +Residual → RMSNorm → SwiGLU → +Residual] x N
      ↓
      RMSNorm → Linear → Softmax

    Features:
      - Pre-Norm (mais estavel que Post-Norm)
      - RMSNorm (mais rapido que LayerNorm)
      - SwiGLU (melhor que GELU/ReLU)
      - RoPE (melhor que positional encoding fixo)
      - Causal Mask (nao vê o futuro)
      - KV Cache (geracao rapida)
      - Top-p/Top-k sampling
      - Repetition penalty
    """

    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=6,
                 ffn_dim=768, dropout=0.1, max_len=512, use_rope=True, use_rmsnorm=True):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.token_emb = Embedding(vocab_size, embed_dim)
        self.drop = Dropout(dropout)

        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(GPTBlock(embed_dim, num_heads, ffn_dim, dropout, max_len, use_rope, use_rmsnorm))

        if use_rmsnorm:
            self.final_norm = RMSNorm(embed_dim)
        else:
            self.final_norm = LayerNorm(embed_dim)

        self.head = Dense(embed_dim, vocab_size)

        self._sync_params()

    def _sync_params(self):
        self.params = {}
        self.grads = {}

        for k, v in self.token_emb.params.items():
            self.params[f"emb.{k}"] = v
            self.grads[f"emb.{k}"] = np.zeros_like(v)

        for i, block in enumerate(self.blocks):
            for k, v in block.params.items():
                self.params[f"blk{i}.{k}"] = v
                self.grads[f"blk{i}.{k}"] = np.zeros_like(v)

        for k, v in self.final_norm.params.items():
            self.params[f"norm.{k}"] = v
            self.grads[f"norm.{k}"] = np.zeros_like(v)

        for k, v in self.head.params.items():
            self.params[f"head.{k}"] = v
            self.grads[f"head.{k}"] = np.zeros_like(v)

    def forward(self, x, use_cache=False, pos_offset=0):
        h = self.token_emb.forward(x)
        h = self.drop.forward(h)

        for block in self.blocks:
            h = block.forward(h, use_cache=use_cache, pos_offset=pos_offset)

        h = self.final_norm.forward(h)
        logits = self.head.forward(h)
        return logits

    def backward(self, grad):
        grad = self.head.backward(grad)
        grad = self.final_norm.backward(grad)

        for block in reversed(self.blocks):
            grad = block.backward(grad)

        grad = self.drop.backward(grad)
        self.token_emb.backward(grad)

        self._copy_grads()

    def _copy_grads(self):
        for k, v in self.token_emb.grads.items():
            key = f"emb.{k}"
            if key in self.grads and v is not None and self.grads[key].shape == v.shape:
                np.copyto(self.grads[key], v)

        for i, block in enumerate(self.blocks):
            for k, v in block.grads.items():
                key = f"blk{i}.{k}"
                if key in self.grads and v is not None and self.grads[key].shape == v.shape:
                    np.copyto(self.grads[key], v)

        for k, v in self.final_norm.grads.items():
            key = f"norm.{k}"
            if key in self.grads and v is not None and self.grads[key].shape == v.shape:
                np.copyto(self.grads[key], v)

    def train(self):
        self.token_emb.train()
        self.drop.train()
        for b in self.blocks:
            b.train()
        self.final_norm.train()
        self.head.train()

    def eval(self):
        self.token_emb.eval()
        self.drop.eval()
        for b in self.blocks:
            b.eval()
        self.final_norm.eval()
        self.head.eval()

    def fit(self, text, epochs=50, batch_size=32, seq_length=128, lr=0.001,
            val_split=0.1, temperature=0.8, top_k=40, top_p=0.9, seed_text=None, gen_every=5):

        dataset = TextDataset(text, seq_length)
        n = int(len(dataset) * (1 - val_split))
        train_data = _Subset(dataset, list(range(n)))
        val_data = _Subset(dataset, list(range(n, len(dataset))))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        loss_fn = CrossEntropyLoss()
        optimizer = AdamW(self.params, self.grads, lr=lr, weight_decay=0.1)

        if seed_text is None:
            seed_text = text[:seq_length]

        def gen_fn(model):
            print(f"\n  --- Geracao (temp={temperature}, top_p={top_p}) ---")
            sample = generate_advanced(model, dataset, seed_text, length=200,
                                        temperature=temperature, top_k=top_k, top_p=top_p)
            print(f"  {sample}")
            print(f"  --- Fim ---")

        trainer = Trainer(self, loss_fn, optimizer, scheduler="warmup_cosine")
        trainer.fit(train_loader, val_loader, epochs=epochs, lr=lr,
                    gen_fn=gen_fn, gen_every=gen_every)
        return trainer.history

    def generate(self, dataset, seed_text, length=500, temperature=0.8, top_k=40, top_p=0.9):
        return generate_advanced(self, dataset, seed_text, length, temperature, top_k, top_p)

    def generate_stream(self, dataset, seed_text, length=500, temperature=0.8, top_k=40, top_p=0.9):
        generate_stream(self, dataset, seed_text, length, temperature, top_k, top_p)

    def clear_cache(self):
        for block in self.blocks:
            block.clear_cache()

    def parameters_count(self):
        return sum(v.size for v in self.params.values())


class _Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    @property
    def seq_length(self):
        return self.dataset.seq_length
