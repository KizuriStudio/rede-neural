#!/bin/bash
# ============================================
# TREINO ULTRA RAPIDO — ~5 min / 1000 epochs
# Otimizado pra Termux (pouca RAM)
# ============================================

cd /storage/emulated/0/Download

python3 rede_neural/train.py \
    --model gpt \
    --file rede_neural/livros \
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
    --save rede_neural/modelo_rapido.npz
