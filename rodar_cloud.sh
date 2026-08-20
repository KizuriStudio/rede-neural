#!/bin/bash
# ============================================
# TREINO EM NUVEM — GitHub Actions
# Otimizado pra Ubuntu Runner (7GB RAM, 2 vCPU)
# ============================================

set -e

# Detecta o diretorio do projeto
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  TREINO EM NUVEM — GitHub Actions                ║"
echo "║  Modelo grande com todos os livros               ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

python3 train.py \
    --model gpt \
    --file livros \
    --embed 128 \
    --hidden 256 \
    --layers 4 \
    --heads 8 \
    --batch 32 \
    --seq-len 128 \
    --epochs 2000 \
    --lr 0.0003 \
    --dropout 0.1 \
    --max-chars 0 \
    --temperature 0.8 \
    --top-p 0.9 \
    --save modelo_cloud.npz

echo ""
echo "Treino concluido!"
ls -lh modelo_cloud.npz modelo_cloud.npz.meta.json 2>/dev/null
