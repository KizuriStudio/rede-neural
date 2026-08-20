#!/bin/bash
# ============================================
# TREINO BALANCEADO — ~30 min / 1000 epochs
# Otimizado pra Termux (pouca RAM)
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 train.py \
    --model gpt \
    --file livros \
    --embed 32 \
    --hidden 64 \
    --layers 2 \
    --heads 2 \
    --batch 8 \
    --seq-len 64 \
    --epochs 1000 \
    --lr 0.0005 \
    --dropout 0.0 \
    --max-chars 300000 \
    --temperature 0.8 \
    --top-p 0.9 \
    --save modelo.npz
