# 🧠 Rede Neural — Gerador de Texto

Framework 100% NumPy de geração de texto com arquitetura GPT Decoder.

## Arquitetura

- **GPT Decoder** estilo GPT-2/LLaMA
- **RMSNorm** (mais rápido que LayerNorm)
- **SwiGLU** (melhor que GELU/ReLU)
- **RoPE** (positional encoding rotacional)
- **Causal Attention** com máscara
- **KV Cache** para geração rápida
- **Top-p / Top-k Sampling** com repetition penalty

## Treino

### Local (Termux)
```bash
# Rápido (~5 min)
bash rede_neural/rodar_rapido.sh

# Balanceado (~30 min)
bash rede_neural/rodar.sh
```

### Nuvem (GitHub Actions)
O workflow `.github/workflows/train.yml` treina automaticamente:
- **5000 epochs** com modelo grande (embed=128, hidden=256, 4 layers)
- **Release automática** do modelo treinado
- Roda em ~3-6 horas no GitHub Actions

### Parâmetros do treino nuvem
| Param | Valor |
|-------|-------|
| Embed dim | 128 |
| Hidden dim | 256 |
| Layers | 4 |
| Heads | 8 |
| Batch size | 32 |
| Seq length | 128 |
| Epochs | 5000 |
| LR | 0.0003 |

## Uso

### Gerar texto
```python
from rede_neural.train import load_model

model, dataset = load_model("rede_neural/modelo_cloud.npz")
text = model.generate(dataset, "O", length=500, temperature=0.8)
print(text)
```

### Modo interativo
```bash
python rede_neural/train.py --load rede_neural/modelo_cloud.npz -i
```

## Dados

8 livros da literatura brasileira:
- Dom Casmurro (Machado de Assis)
- Iracema (José de Alencar)
- Memórias Póstumas (Machado de Assis)
- O Cortiço (Aluísio Azevedo)
- Os Lusíadas (Camões)
- Viagens na Minha Terra (Almeida Garrett)
- Humus (Raul Brandão)
- A Escrava Isaura (Bernardo Guimarães)

## Estrutura

```
rede_neural/
├── train.py              # Script principal
├── rodar_rapido.sh       # Treino rápido (Termux)
├── rodar.sh              # Treino balanceado (Termux)
├── rodar_cloud.sh        # Treino para nuvem
├── layers/               # Camadas da rede
│   ├── embedding.py
│   ├── dense.py
│   ├── rmsnorm.py
│   ├── swiglu.py
│   ├── rope.py
│   ├── causal_attention.py
│   ├── dropout.py
│   └── sequential.py
├── models/
│   ├── gpt_decoder.py    # GPT Decoder
│   └── text_generator.py # LSTM / Transformer
├── losses/
│   └── losses.py
├── optimizers/
│   └── optimizers.py     # AdamW
├── utils/
│   ├── data.py           # TextDataset + DataLoader
│   ├── training.py       # Trainer + LR Schedulers
│   ├── sampling.py       # Top-p / Top-k sampling
│   └── text.py
└── livros/               # Dados de treino
```

## Licença

MIT
