from langchain_community.document_loaders import PyPDFLoader
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os
from typing import List, Set
from tqdm import tqdm
from utils.text_splitter.medical_splitter import AdaptiveMedicalSplitter
from embedding_model import embedding_model

def get_pdf_files(pdf_dir: str) -> Set[str]:
    """获取目录下的所有PDF文件"""
    return {f for f in os.listdir(pdf_dir) if f.endswith('.pdf')}

def load_processed_files(filepath: str) -> Set[str]:
    """加载已处理的文件列表"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())
    except FileNotFoundError:
        return set()

def save_processed_files(filepath: str, files: Set[str]):
    """保存已处理的文件列表"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for file in files:
            f.write(file + '\n')

def print_knowledge_base_info(current_files, new_files=None):
    """打印知识库信息"""
    print("\n" + "="*50)
    print("知识库包含以下文件：")
    for file in sorted(current_files):  # 按字母顺序排
        status = "[新文件]" if new_files and file in new_files else "[已加载]"
        print(f"  {status} {file}")
    print("="*50 + "\n")

def init_milvus_collection():
    """初始化 Milvus 集合"""
    collection_name = "medical_knowledge_base"
    
    # 创建集合
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="chunk_index", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="chunk_total", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="chunk_size", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="chunk_overlap", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1792)
    ]
    schema = CollectionSchema(fields, "Medical knowledge base")
    collection = Collection(collection_name, schema)
    
    # 创建索引
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_PQ",
        "params": {"nlist": 1024, "m": 16},
        "device": "cuda"
    }
    collection.create_index("embedding", index_params)
    collection.load()
    
    return collection

def load_pdfs(pdf_dir):
    model_name = "TencentBAC/Conan-embedding-v1"
    print(f"（{model_name}）开始加载embedding模型...")
    embeddings = embedding_model
    print("embedding模型加载完成")

    connections.connect("default", host="localhost", port="19530")
    collection_name = "medical_knowledge_base"

    current_files = get_pdf_files(pdf_dir)
    if not current_files:
        print(f"\n警告: {pdf_dir} 目录下没有找到PDF文件")
        # 如果目录为空，清空整个集合
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
            collection.delete(expr="id >= 0")  # 删除所有数据
            print("已清空向量库中的所有数据")
        return None

    # 获取已处理的文件名和处理删除的文件
    processed_files = set()
    try:
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
            
            # 获取所有已处理的文件
            for entity in collection.query(expr='id >= 0', output_fields=['source']):
                processed_files.add(entity['source'])
            
            # 检查并处理已删除的文件
            deleted_files = processed_files - current_files
            if deleted_files:
                print("\n检测到以下文件已从目录中删除：")
                for file in deleted_files:
                    print(f"  - {file}")
                    expr = f'source == "{file}"'
                    collection.delete(expr)
                print("已从向量库中删除对应数据")
                # 更新processed_files，移除已删除的文件
                processed_files = processed_files - deleted_files
                
    except Exception as e:
        print(f"捕获到异常: {e}")
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        print("Milvus集合已删除，将重新创建")
        collection = init_milvus_collection()
        processed_files = set()

    new_files = current_files - processed_files

    if new_files:
        if not utility.has_collection(collection_name):
            collection = init_milvus_collection()
        else:
            collection = Collection(collection_name)
            collection.load()

        print_knowledge_base_info(current_files, new_files)

        splitter = AdaptiveMedicalSplitter()
        documents = []

        print("\n开始处理文档...")
        batch_size = 1
        for i in range(0, len(new_files), batch_size):
            batch_files = list(new_files)[i:i + batch_size]
            for file in tqdm(batch_files, desc="加载PDF"):
                loader = PyPDFLoader(os.path.join(pdf_dir, file))
                raw_docs = loader.load()

                full_text = "\n\n".join(doc.page_content for doc in raw_docs)
                processed_docs = splitter.split_document(
                    full_text,
                    metadata={"source": file}
                )
                documents.extend(processed_docs)

        if documents:
            print(f"\n总计分割为 {len(documents)} 个文本块")
            print("将文本块输入Milvus...")
            data = [
                [doc.metadata['source'] for doc in documents],
                [doc.metadata['chunk_index'] for doc in documents],
                [doc.metadata['chunk_total'] for doc in documents],
                [doc.metadata['chunk_size'] for doc in documents],
                [doc.metadata['chunk_overlap'] for doc in documents],
                [doc.page_content for doc in documents],
                [embeddings.embed_query(doc.page_content) for doc in documents]
            ]
            collection.insert(data)

    else:
        print("未检测到PDF文件变化，使用现有Milvus向量库...")
        collection = Collection(collection_name)
        collection.load()
        print_knowledge_base_info(current_files)

    return collection

def load_pdf(pdf_path: str, collection, embeddings):
    """向量化指定的PDF文件"""
    print(f"开始处理文档: {pdf_path}...")
    splitter = AdaptiveMedicalSplitter()
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    full_text = "\n\n".join(doc.page_content for doc in raw_docs)
    processed_docs = splitter.split_document(
        full_text,
        metadata={"source": os.path.basename(pdf_path)}  # 使用文件名作为source
    )
    documents = processed_docs

    print(f"\n总计分割为 {len(documents)} 个文本块")

    print("将文本块输入Milvus...")
    data = [
        [doc.metadata['source'] for doc in documents],
        [doc.metadata['chunk_index'] for doc in documents],
        [doc.metadata['chunk_total'] for doc in documents],
        [doc.metadata['chunk_size'] for doc in documents],
        [doc.metadata['chunk_overlap'] for doc in documents],
        [doc.page_content for doc in documents],
        [embeddings.embed_query(doc.page_content) for doc in documents]
    ]

    collection.insert(data)
    print("创建索引...")
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_PQ",
        "params": {"nlist": 1024, "m": 16},
        "device": "cuda"
    }
    collection.create_index("embedding", index_params)
    collection.load()
    print(f"文档 {pdf_path} 向量化完成")

    return collection