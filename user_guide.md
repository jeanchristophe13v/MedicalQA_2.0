# 医学知识库问答系统使用指南

## 1. 项目简介

本项目旨在搭建一个基于本地医学教材的知识库问答系统。用户可以通过提问的方式，从知识库中获取相关的医学知识。系统使用Milvus向量数据库存储知识，并利用Langchain框架和Gemini模型实现问答功能。

## 2. 环境要求

*   **操作系统**: Windows 11 (已测试), 理论上支持其他安装了Docker的操作系统
*   **Python**: 3.11 (已测试), 理论上支持Python 3.7及以上版本
*   **Docker**: 已安装并配置好Docker和Docker Compose

## 3. 安装步骤

1.  **克隆项目代码**:

    ```bash
    git clone (仓库地址)  # 将 (仓库地址) 替换为实际的项目仓库地址
    cd medicalQA_large-gte-zh
    ```

2.  **创建并激活虚拟环境**:

    ```bash
    python -m venv venv2
    venv2\Scripts\activate
    ```
    
3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
    如果安装过程中出现错误, 可以尝试使用`pip install -r requirements.txt --force-reinstall` 强制重新安装

## 4. 配置

### 4.1 Milvus配置

本项目使用Milvus作为向量数据库。您需要使用Docker Compose启动Milvus服务。

1.  **启动Milvus**:

    在项目根目录下执行以下命令：

    ```bash
    docker compose up -d
    ```

    这将根据`docker-compose.yml`文件中的配置启动Milvus及其依赖服务（etcd和minio）。

### 4.2 Gemini API Key配置

本项目使用Google Gemini模型进行问答。您需要获取Gemini API Key并配置为环境变量。

1.  **获取Gemini API Key**:

    访问Google AI Studio (https://ai.google.dev/) 并获取您的API Key。

2.  **配置环境变量**:

    *   **方法一 (推荐)**：在项目根目录下创建一个名为`.env`的文件，并在其中添加以下内容：

        ```
        GEMINI_API_KEY=您的API密钥
        ```
        将 "您的API密钥" 替换为您实际的Gemini API Key。

    *   **方法二**：直接在系统的环境变量中设置`GEMINI_API_KEY`。

### 4.3 PDF文件配置

将您需要添加到知识库的PDF文件（例如医学教材）放入`data`目录下。

## 5. 使用

1.  **运行程序**:

    确保您已经激活了`venv2`虚拟环境，然后在项目根目录下执行以下命令：

    ```bash
    venv2/Scripts/python main.py
    ```

2.  **与程序交互**:

    *   程序启动后，会打印欢迎信息和加载提示。
    *   首次运行或`data`目录下有新的PDF文件时，程序会自动处理PDF文件，构建或更新Milvus向量数据库。这可能需要一些时间，请耐心等待。
    *   加载完成后，您可以在终端中输入问题，程序会从知识库中检索相关信息并生成回答。
    *   输入`q`或`quit`或`exit`可以退出程序。
    *   输入`clear`可以清屏。

## 6. 注意事项

*   PDF文件的处理和向量化可能需要较长时间，请耐心等待。
*   如果程序运行过程中出现错误，请仔细阅读错误信息，并根据提示进行操作。
*   建议定期检查`requirements.txt`文件，并使用`pip install -r requirements.txt --upgrade`更新依赖。
*   Milvus默认监听`localhost:19530`端口。

## 7. 高级配置 (可选)

*   **自定义文本分割策略**: 您可以通过修改`utils/text_splitter/medical_splitter.py`文件中的`AdaptiveMedicalSplitter`类来自定义文本分割策略。
*   **自定义Prompt**: 您可以通过修改`chat_agent.py`文件中的`prompt`变量来自定义提问模板。