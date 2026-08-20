#!/bin/bash
# ============================================
# TREINO BALANCEADO — ~30 min / 1000 epochs
# Otimizado pra Termux (pouca RAM)
# ============================================

cd /storage/emulated/0/Download

python3 rede_neural/train.py \
    --model gpt \
    --file rede_neural/livros \
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
    --save rede_neural/modelo.npz
