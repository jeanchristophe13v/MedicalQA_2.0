from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os
from typing import List
from tqdm import tqdm
from utils.text_splitter.medical_splitter import AdaptiveMedicalSplitter

def get_pdf_files(pdf_dir):
    """获取目录下的所有PDF文件"""
    return {f for f in os.listdir(pdf_dir) if f.endswith('.pdf')}

def print_knowledge_base_info(current_files, new_files=None):
    """打印知识库信息"""
    print("\n" + "="*50)
    print("知识库包含以下文件：")
    for file in sorted(current_files):  # 按字母顺序排
        status = "[新文件]" if new_files and file in new_files else "[已加载]"
        print(f"  {status} {file}")
    print("="*50 + "\n")

def load_pdfs(pdf_dir):
    print("开始加载embedding模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="TencentBAC/Conan-embedding-v1",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
        cache_folder="./embeddings_cache"
    )
    print("embedding模型加载完成")

    current_files = get_pdf_files(pdf_dir)
    if not current_files:
        print(f"\n警告: {pdf_dir} 目录下没有找到PDF文件")
        return None

    connections.connect("default", host="localhost", port="19530")
    collection_name = "medical_knowledge_base"

    # 获取已处理的文件名
    processed_files = set()
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        # Milvus 2.4+ 需要先 load 才能 query
        collection.load()
        # 假设source字段存储了文件名
        for entity in collection.query(expr='id >= 0', output_fields=['source']):
            processed_files.add(entity['source'])

    if set(current_files) != processed_files:
        if utility.has_collection(collection_name):
            print("检测到PDF文件变化，删除旧的Milvus集合...")
            utility.drop_collection(collection_name)

        print("\n创建新的Milvus向量库...")
        print("定义Milvus集合模式...")
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

        print_knowledge_base_info(current_files)
        splitter = AdaptiveMedicalSplitter()
        documents = []

        print("\n开始处理文档...")
        batch_size = 1  # 每次加载一个文件
        for i in range(0, len(current_files), batch_size):
            batch_files = list(current_files)[i:i + batch_size]
            for file in tqdm(batch_files, desc="加载PDF"):
                loader = PyPDFLoader(os.path.join(pdf_dir, file))
                raw_docs = loader.load()

                full_text = "\n\n".join(doc.page_content for doc in raw_docs)
                processed_docs = splitter.split_document(
                    full_text,
                    metadata={"source": file}
                )
                documents.extend(processed_docs)

        print(f"\n总计分割为 {len(documents)} 个文本块")

        print("将数据插入Milvus...")
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
        print("Milvus向量化完成")
        return collection
    else:
        print("未检测到PDF文件变化，使用现有Milvus向量库...")
        collection = Collection(collection_name)
        collection.load()
        return collection