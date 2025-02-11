#!/bin/bash

# 停止 Milvus 容器
docker compose down

# 删除 volumes 文件夹
rm -rf volumes

# 启动 Milvus 容器
docker compose up -d

# 删除 data 文件夹中的原有 PDF 文件
rm -rf data/*.pdf

echo "data 文件夹已格式化，请添加新的 PDF 文件。"