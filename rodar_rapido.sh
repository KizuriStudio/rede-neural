#!/bin/bash
# ============================================
# TREINO ULTRA RAPIDO — ~5 min / 2000 epochs
# Otimizado pra Termux (pouca RAM)
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 train.py \
    --model gpt \
    --file livros \
    --embed 16 \
    --hidden 32 \
    --layers 1 \
    --heads 2 \
    --batch 4 \
    --seq-len 32 \
    --epochs 2000 \
    --lr 0.001 \
    --dropout 0.0 \
    --max-chars 200000 \
    --temperature 0.8 \
    --top-p 0.9 \
    --save modelo_rapido.npz
