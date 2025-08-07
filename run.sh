#!/bin/bash

# 确保你在正确的工作目录
cd D:\\github_files  # 这里的路径是你的应用程序所在的文件夹

# 安装依赖 (如果还没有安装的话)
# pip install -r requirements.txt

# 启动 Uvicorn 服务器
uvicorn qwen3_app:app --reload
