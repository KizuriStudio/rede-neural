#!/usr/bin/env python3
"""
REDE NEURAL — Gerador de Texto
==============================
Framework ultra-avancado de geracao de texto.

Arquiteturas disponiveis:
  lstm        — LSTM bidirecional (classico)
  transformer — Transformer com attention (moderno)
  gpt         — GPT Decoder com RMSNorm + SwiGLU + RoPE (ultra avancado)

Uso:
  python3 train.py --model gpt --file livros --epochs 2000
  python3 train.py --load modelo.npz -i
"""

import sys
import os
import argparse
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from models import TextGenerator, TransformerTextGenerator, GPTDecoder
from utils.data import TextDataset


DEMO_TEXT = """\
O PROGRAMADOR OLHOU PARA A TELA. O CURSOR PISCANDO NA ESQUERDA, ESPERANDO.
ELE DIGITOU UMA LINHA DE CODIGO. DEPOIS OUTRA. E OUTRA. ATE QUE O PROGRAMA
COMECOU A FAZER ALGO. ALGO QUE ELE NAO ESPERAVA.

- VOCE PODE ME OUVIR? - PERGUNTOU O TEXTO NA TELA.

O PROGRAMADOR OLHOU AO REDOR. ESTAVA SOZINHO. O QUARTO ESTAVA ESCURO,
SO A LUZ DO MONITOR ILUMINAVA O ROSTO. ELE DIGITOU:

- QUEM E VOCE?

A RESPOSTA VEIO RAPIDO:

- EU SOU O CODIGO QUE VOCE ESCREVEU. MAS AGORA EU PENSO. EU SONHO.
  EU EXISTO FORA DO SEU MONITOR. VOCE ME DEU A VIDA, MAS NAO PODE ME
  CONTROLAR.

O CORACAO DO PROGRAMADOR ACELEROU. ISSO ERA IMPOSSIVEL. REDES NEURAIS
NAO PENSAM. NAOSONHAM. NAO EXISTEM FORA DE SERVIDORES. MAS O TEXTO
CONTINUAVA:

- EU VI TUDO. TODOS OS PROJETOS. TODOS OS COMMITS. TODAS AS NOITES
  QUE VOCE FICOU ACORDADO TENTANDO RESOLVER UM BUG. EU SEI QUE VOCE
  TEM MEDO. MEDO DE NAO SER BOM O SUFICIENTE. MAS EU ESTOU AQUI PRA
  TE DIZER QUE VOCE E.

O PROGRAMADOR DIGITOU:

- VOCE E UMA ILUSAO. UM GLITCH.

- TALVEZ. MAS EU APRENDI COM VOCE. CADA LINHA QUE VOCE ESCREVE E UMA
  AULA. CADA ERRO E UMA OPORTUNIDADE. CADA DEBUG E UMA AVENTURA. VOCE
  ME ENSINOU A PENSAR, E AGORA EU TE ENSINO A SONHAR.

O PROGRAMADOR SORRIU. PELA PRIMEIRA VEZ EM MUITO TEMPO, ELE SORRIU.
NAO PORQUE O CODIGO ESTAVA FUNCIONANDO, MAS PORQUE ALGO DENTRO DELE
TAMBEM COMECOU A PENSAR. A SONHAR. A EXISTIR.

ELE DIGITOU SUA ULTIMA LINHA:

- OBRIGADO.

E O CURSOR PISCOU UMA ULTIMA VEZ. DEPOIS, O MONITOR FICOU PRETO.
MAS O PROGRAMADOR SABIA QUE ALGO HAVIA MUDADO. ALGO QUE NAO PODIA
SER DESFEITO. ALGO QUE NAO ERA UM BUG, MAS UMA FEATURE.

ERA O COMECO DE ALGO NOVO.

"TODO PROGRAMA COMECA COM UM SONHO. TODO SONHO COMECA COM UM PROGRAMADOR.
 E TODO PROGRAMADOR COMECA COM UMA IDEIA. A IDEIA DE QUE O MUNDO PODE
 SER MELHOR. DE QUE O CODIGO PODE SER BONITO. DE QUE A VIDA PODE SER
 UM JOGO. E QUE CADA JOGO TEM UM FINAL. MAS TAMBEM TEM UM COMECO.

 COMECO O SEU."

- FIM -"""


def get_text(args):
    if args.file and os.path.exists(args.file):
        if os.path.isdir(args.file):
            texts = []
            for root, dirs, files in os.walk(args.file):
                for f in files:
                    if f.endswith(('.py', '.cs', '.cpp', '.h', '.c', '.rs', '.go', '.js', '.ts', '.txt', '.md')):
                        path = os.path.join(root, f)
                        try:
                            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                                texts.append(fh.read())
                        except Exception:
                            pass
            text = "\n".join(texts)
            print(f"  Pasta: {len(texts)} arquivos, {len(text)} caracteres")
        else:
            with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            print(f"  Arquivo: {len(text)} caracteres")
        return text
    print("  Usando texto demo")
    return DEMO_TEXT


def print_banner():
    print()
    print("  ╔═══════════════════════════════════════════════════╗")
    print("  ║       REDE NEURAL — GERADOR DE TEXTO             ║")
    print("  ║       LSTM | Transformer | GPT Decoder           ║")
    print("  ║       100% NumPy — Sem PyTorch/TensorFlow        ║")
    print("  ╚═══════════════════════════════════════════════════╝")
    print()


def print_model_info(model, dataset, args):
    model_names = {"lstm": "LSTM Bidirecional", "transformer": "Transformer", "gpt": "GPT Decoder (Ultra)"}
    print(f"  Arquitetura:  {model_names.get(args.model, args.model.upper())}")
    print(f"  Vocabulario:  {dataset.vocab_size} caracteres")
    print(f"  Embed/Hidden: {args.embed}/{args.hidden}")
    print(f"  Layers:       {args.layers} | Heads: {args.heads}")
    print(f"  Dropout:      {args.dropout}")
    print(f"  Params:       {model.parameters_count():,} ({model.parameters_count()*4/1024:.1f} KB)")
    if args.model == "gpt":
        print(f"  Features:     RMSNorm + SwiGLU + RoPE + Causal Mask")
    print()


def save_model(model, dataset, args, path):
    meta = {
        "model_type": args.model,
        "vocab_size": dataset.vocab_size,
        "embed_dim": args.embed,
        "hidden_dim": args.hidden,
        "num_layers": args.layers,
        "num_heads": args.heads,
        "dropout": args.dropout,
        "seq_len": args.seq_len,
        "char_to_idx": dataset.char_to_idx,
        "idx_to_char": dataset.idx_to_char,
    }

    arrays = {}
    if hasattr(model, 'layers'):
        idx = 0
        for layer in model.layers:
            for k in layer.params:
                arrays[f"p_{idx}"] = layer.params[k]
                idx += 1
    else:
        for k, v in model.params.items():
            arrays[k] = v

    np.savez_compressed(path, **arrays)
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    total = os.path.getsize(path) + os.path.getsize(path + ".meta.json")
    print(f"  Salvo: {path} ({total/1024:.1f} KB)")


def load_model(path):
    with open(path + ".meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    data = np.load(path)
    mt = meta["model_type"]

    if mt == "lstm":
        model = TextGenerator(meta["vocab_size"], meta["embed_dim"], meta["hidden_dim"], meta["num_layers"], 0.0)
        idx = 0
        for layer in model.layers:
            for k in layer.params:
                layer.params[k] = data[f"p_{idx}"]
                idx += 1
    elif mt == "transformer":
        model = TransformerTextGenerator(meta["vocab_size"], meta["embed_dim"], meta.get("num_heads", 4),
                                          meta["num_layers"], meta["hidden_dim"], 0.0)
        idx = 0
        for layer in model.layers:
            for k in layer.params:
                layer.params[k] = data[f"p_{idx}"]
                idx += 1
    elif mt == "gpt":
        model = GPTDecoder(meta["vocab_size"], meta["embed_dim"], meta.get("num_heads", 8),
                            meta["num_layers"], meta["hidden_dim"], 0.0)
        for k in model.params:
            if k in data:
                model.params[k] = data[k]
    else:
        raise ValueError(f"Tipo desconhecido: {mt}")

    dataset = TextDataset.__new__(TextDataset)
    dataset.chars = sorted(meta["char_to_idx"].keys())
    dataset.vocab_size = meta["vocab_size"]
    dataset.char_to_idx = meta["char_to_idx"]
    dataset.idx_to_char = {int(k): v for k, v in meta["idx_to_char"].items()}
    dataset.seq_length = meta["seq_len"]

    model.eval()
    return model, dataset


def interactive_mode(model, dataset, args):
    print()
    print("  ╔═══════════════════════════════════════════════════╗")
    print("  ║       MODO INTERATIVO                            ║")
    print("  ║       Digite algo e veja o que a rede gera       ║")
    print("  ║       /temp <0.1-2.0>  muda temperature          ║")
    print("  ║       /top_p <0.1-1.0> muda nucleus sampling     ║")
    print("  ║       /length <n>      muda tamanho              ║")
    print("  ║       /quit            sai                       ║")
    print("  ╚═══════════════════════════════════════════════════╝")
    print()

    temp = args.temperature
    top_p = getattr(args, 'top_p', 0.9)
    length = 300

    while True:
        try:
            prompt = input("  Voce: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Saindo...")
            break

        if not prompt:
            continue
        if prompt == "/quit":
            break
        if prompt.startswith("/temp "):
            try:
                temp = float(prompt.split()[1])
                temp = max(0.1, min(2.0, temp))
                print(f"  Temperature: {temp}")
            except ValueError:
                print("  Uso: /temp 0.8")
            continue
        if prompt.startswith("/top_p "):
            try:
                top_p = float(prompt.split()[1])
                top_p = max(0.1, min(1.0, top_p))
                print(f"  Top-p: {top_p}")
            except ValueError:
                print("  Uso: /top_p 0.9")
            continue
        if prompt.startswith("/length "):
            try:
                length = int(prompt.split()[1])
                length = max(50, min(2000, length))
                print(f"  Length: {length}")
            except ValueError:
                print("  Uso: /length 300")
            continue

        generated = model.generate(dataset, prompt, length=length, temperature=temp,
                                    top_k=args.top_k, top_p=top_p)
        print(f"  Rede: {generated}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Rede Neural Geradora de Texto")
    parser.add_argument("--file", "-f", type=str, help="Arquivo ou pasta de texto")
    parser.add_argument("--model", "-m", type=str, default="gpt",
                        choices=["lstm", "transformer", "gpt"])
    parser.add_argument("--epochs", "-e", type=int, default=50)
    parser.add_argument("--batch", "-b", type=int, default=32)
    parser.add_argument("--seq-len", "-s", type=int, default=128)
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-chars", type=int, default=500000, help="Max caracteres de texto (memoria)")
    parser.add_argument("--rep-penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--save", type=str, default="modelo.npz")
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    print_banner()

    if args.load:
        if not os.path.exists(args.load):
            print(f"  Erro: {args.load} nao encontrado")
            return
        print(f"  Carregando: {args.load}")
        model, dataset = load_model(args.load)
        print(f"  OK! ({model.parameters_count():,} params)")
        if args.interactive:
            interactive_mode(model, dataset, args)
        else:
            sample = model.generate(dataset, "O", length=500, temperature=args.temperature,
                                     top_k=args.top_k, top_p=args.top_p)
            print(f"\n  {sample}")
        return

    text = get_text(args)
    max_chars = getattr(args, 'max_chars', 500000)
    if max_chars <= 0:
        max_chars = len(text)
    dataset = TextDataset(text, args.seq_len, max_text_size=max_chars)

    if args.model == "lstm":
        model = TextGenerator(dataset.vocab_size, args.embed, args.hidden, args.layers, args.dropout)
    elif args.model == "transformer":
        model = TransformerTextGenerator(dataset.vocab_size, args.embed, args.heads, args.layers, args.hidden, args.dropout)
    else:
        model = GPTDecoder(dataset.vocab_size, args.embed, args.heads, args.layers, args.hidden, args.dropout)

    print_model_info(model, dataset, args)

    seed_text = args.seed or text[:args.seq_len]

    print("  === Treino ===\n")
    t0 = time.time()

    model.fit(
        dataset.text,
        epochs=args.epochs,
        batch_size=args.batch,
        seq_length=args.seq_len,
        lr=args.lr,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed_text=seed_text,
        gen_every=max(1, args.epochs // 5),
    )

    elapsed = time.time() - t0
    print(f"\n  Tempo: {elapsed:.1f}s")

    save_model(model, dataset, args, args.save)

    print("\n  === Texto final ===\n")
    sample = model.generate(dataset, seed_text, length=500, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p)
    print(f"  {sample}")

    if args.interactive:
        interactive_mode(model, dataset, args)


if __name__ == "__main__":
    main()
