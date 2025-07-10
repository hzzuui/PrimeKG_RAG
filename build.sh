#!/bin/bash

# 啟用錯誤提示
set -e

echo "🔧 建立 Python 虛擬環境..."
python3 -m venv venv

echo "📦 啟動虛擬環境..."
source venv/bin/activate

echo "🔄 升級 pip..."
pip install --upgrade pip

echo "🔥 安裝 PyTorch (CPU 版本)..."
pip install torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

echo "📚 安裝其他依賴套件..."
pip install -r requirements.txt

echo "✅ 完成！虛擬環境與依賴已建立。"