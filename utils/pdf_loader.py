from langchain_community.document_loaders import PyPDFLoader
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os
from typing import List, Set, Dict
import re
from tqdm import tqdm
from utils.text_splitter.medical_splitter import AdaptiveMedicalSplitter
from embedding_model import embedding_model

def get_pdf_files(pdf_dir: str, specific_files: List[str] = None) -> Set[str]:
    """获取目录下的PDF文件"""
    if specific_files:
        # 验证指定的文件是否存在
        existing_files = {f for f in os.listdir(pdf_dir) if f.endswith('.pdf')}
        valid_files = set()
        for file in specific_files:
            if file in existing_files:
                valid_files.add(file)
            else:
                print(f"警告: 文件 '{file}' 在 {pdf_dir} 目录中未找到")
        return valid_files
    else:
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

def get_collection_name(filename: str) -> str:
    """根据PDF文件名生成对应的collection名称"""
    # 移除.pdf后缀
    base_name = filename.replace('.pdf', '')
    
    # 将中文字符转换为拼音(需要安装pypinyin: pip install pypinyin)
    from pypinyin import lazy_pinyin
    pinyin_list = lazy_pinyin(base_name)
    pinyin_name = '_'.join(pinyin_list)
    
    # 移除所有特殊字符,只保留字母、数字和下划线
    import re
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', pinyin_name)
    
    # 确保名称不以数字开头
    if clean_name[0].isdigit():
        clean_name = f"kb_{clean_name}"
        
    return f"medical_kb_{clean_name}"

def init_collection(collection_name: str):
    """为单个PDF文件初始化collection"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_index", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="chunk_total", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="chunk_size", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="chunk_overlap", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1792)
    ]
    schema = CollectionSchema(fields, f"Collection for {collection_name}")
    collection = Collection(collection_name, schema)
    
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_PQ",
        "params": {"nlist": 1024, "m": 16},
        "device": "cuda"
    }
    collection.create_index("embedding", index_params)
    collection.load()
    return collection

def print_step(text):
    """打印步骤信息，确保清除loading残留"""
    from main import print_with_loading_clear
    print_with_loading_clear(text)

def load_pdf(pdf_path: str, embeddings) -> Collection:
    """处理单个PDF文件"""
    filename = os.path.basename(pdf_path)
    collection_name = get_collection_name(filename)
    
    print_step(f"\n开始处理文档: {filename}")
    splitter = AdaptiveMedicalSplitter()
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    
    print_step("   1. 分割文档")
    full_text = "\n\n".join(doc.page_content for doc in raw_docs)
    documents = splitter.split_document(full_text)
    chunks_count = len(documents)
    print_step(f"      → 得到 {chunks_count} 个文本块")
    
    # 创建新的collection
    print_step("   2. 初始化向量集合")
    collection = init_collection(collection_name)
    
    # 插入数据
    if documents:
        print_step("   3. 开始向量化")
        with tqdm(total=len(documents), desc="      进度", ncols=70) as pbar:
            data = [
                [doc.metadata['chunk_index'] for doc in documents],
                [doc.metadata['chunk_total'] for doc in documents],
                [doc.metadata['chunk_size'] for doc in documents],
                [doc.metadata['chunk_overlap'] for doc in documents],
                [doc.page_content for doc in documents],
                []  # 预分配embedding列表
            ]
            
            # 分批处理向量化
            batch_size = 4  # 可以根据需要调整批大小
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                embeddings_batch = [embeddings.embed_query(doc.page_content) for doc in batch]
                data[5].extend(embeddings_batch)
                pbar.update(len(batch))
        
        collection.insert(data)
    
    return collection

def load_pdfs(pdf_dir: str, specific_files: List[str] = None) -> Dict[str, Collection]:
    """加载PDF文件到向量库"""
    print_step("\n初始化中...")
    print_step("1. 加载embedding模型 (TencentBAC/Conan-embedding-v1)")
    embeddings = embedding_model
    print_step("   ✓ 模型加载完成")

    print_step("\n2. 连接Milvus数据库")
    connections.connect("default", host="localhost", port="19530")
    print_step("   ✓ 数据库连接成功")
    
    # 获取需要处理的文件
    current_files = get_pdf_files(pdf_dir, specific_files)
    if not current_files:
        print_step(f"\n警告: 没有找到要处理的PDF文件")
        return {}

    print_step("\n3. 扫描PDF文件和向量库")
    # 获取所有collection名称
    existing_collections = {}
    new_files = set()
    collections_to_delete = set()
    
    # 获取所有已存在的collections
    all_collections = utility.list_collections()
    
    # 检查每个数据库中的collection
    for collection_name in all_collections:
        if collection_name.startswith("medical_kb_"):
            # 尝试从collection名称反推文件名
            # 这里需要添加一个新函数来实现这个功能
            original_filename = get_original_filename(collection_name)
            if original_filename:
                # 如果文件不在当前目录中，加入待删除列表
                if original_filename not in current_files:
                    collections_to_delete.add(collection_name)
                    print_step(f"   ! 发现孤立向量库: {original_filename}")
    
    # 删除不存在文件对应的collections
    if collections_to_delete:
        print_step("\n清理孤立向量库中")
        for collection_name in collections_to_delete:
            utility.drop_collection(collection_name)
            print_step(f"   ✓ 已删除: {collection_name}")
    
    # 处理当前文件
    print_step("\n4. 处理当前文件")
    for file in current_files:
        collection_name = get_collection_name(file)
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
            existing_collections[file] = collection
        else:
            new_files.add(file)

    if not new_files:
        print_step("   ✓ 所有文件已加载")
    else:
        for file in new_files:
            print_step(f"   → 待处理: {file}")
    
    # 处理新文件
    if new_files:
        print_step("\n5. 处理新文件")
        for file in new_files:
            pdf_path = os.path.join(pdf_dir, file)
            print_step(f"\n处理文件: {file}")
            collection = load_pdf(pdf_path, embeddings)
            existing_collections[file] = collection
            print_step(f"   ✓ 向量化完成")
    
    print_step("\n" + "="*50 +"\n")
    print_step("知识库加载完成，包含以下文件：")
    for file in sorted(current_files):
        status = "[新文件]" if file in new_files else "[已加载]"
        print_step(f"  {status} {file}")
    print_step("\n"+"="*50)
    
    return existing_collections

def get_original_filename(collection_name: str) -> str:
    """从collection名称反推原始文件名"""
    # 移除前缀
    if not collection_name.startswith("medical_kb_"):
        return None
    name = collection_name[len("medical_kb_"):]
    
    # 将拼音转换回可能的原始文件名
    # 由于拼音转换是单向的，这里我们只能通过现有文件来匹配
    def normalize_name(s):
        return re.sub(r'[^a-zA-Z0-9_]', '_', s.lower())
    
    # 获取当前目录下所有PDF文件
    pdf_files = {f for f in os.listdir("data") if f.endswith('.pdf')}
    
    # 对每个文件名生成collection名称，找到匹配的
    for pdf_file in pdf_files:
        if get_collection_name(pdf_file)[len("medical_kb_"):] == name:
            return pdf_file
    
    return None