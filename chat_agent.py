from dotenv import load_dotenv
import os
import re
import jieba

# 初始化环境变量
load_dotenv()

class ChatAgent:
    def __init__(self, pdf_dir, specific_files=None):
        # 延迟导入其他模块
        from main import print_with_loading_clear
        print_with_loading_clear("正在加载必要组件...")
        
        import google.generativeai as genai
        import jieba
        from pymilvus import connections, Collection, utility
        from utils.pdf_loader import load_pdfs
        from embedding_model import embedding_model
        
        # 设置日志级别
        import logging
        jieba.setLogLevel(logging.WARNING)
        
        try:
            connections.connect("default", host="localhost", port="19530")
        except Exception as e:
            print("正在连接 Milvus 服务器...")
            connections.connect("default", host="localhost", port="19530")

        # Gemini API 配置
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key, transport="rest")

        # 生成参数配置
        self.model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-thinking-exp-01-21',
            generation_config={
                "temperature": 0.4,
                "top_p": 1,
                "top_k": 40,
                "max_output_tokens": 65536,
            }
        )
        # self.model = genai.GenerativeModel(
        #     model_name='gemini-2.0-pro-exp',
        #     generation_config={
        #         "temperature": 0.2,
        #         "top_p": 1,
        #         "top_k": 40,
        #         "max_output_tokens": 8192,
        #     }
        # )

        # 初始化或加载知识库
        self.collections = {}
        self.pdf_dir = pdf_dir
        
        # 加载PDF文件
        self.collections = load_pdfs(pdf_dir, specific_files)
        if not self.collections:
            print("警告: 未能加载任何PDF文件")
            
        # 初始化embedding模型
        self.embeddings = embedding_model
        
        self.chat_history = []
        self.max_history = 10

    def chat(self, query):
        try:
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 16}
            }
            
            # 在所有加载的collections中搜索
            all_results = []
            query_embedding = self.embeddings.embed_query(query)
            
            for filename, collection in self.collections.items():
                results = collection.search(
                    [query_embedding], 
                    "embedding",
                    search_params,
                    limit=4,  # 每个文件取前4个最相关的结果
                    output_fields=["chunk_index", "chunk_total", "content"]
                )
                
                for hits in results:
                    for hit in hits:
                        doc = {
                            'page_content': hit.get('content'),
                            'metadata': {
                                'source': filename,
                                'chunk_index': hit.get('chunk_index'),
                                'chunk_total': hit.get('chunk_total'),
                                'score': hit.score  # 添加相似度分数
                            }
                        }
                        all_results.append(doc)
            
            # 根据相似度分数排序,取最相关的内容
            all_results.sort(key=lambda x: x['metadata']['score'])
            filtered_docs = all_results[:12]  # 取总体最相关的12个结果

            # Milvus返回的结果处理
            context = ""
            current_length = 0
            max_length = min(8000, 3000 + len(query) * 10)
            
            for doc in filtered_docs:
                content = doc['page_content']
                if current_length + len(content) > max_length:
                    break
                context += f"\n\n{content}"
                current_length += len(content)
            
            # 构建提示词
            history_text = ""
            if self.chat_history:
                history_text = "\n历史对话：\n" + "\n".join(
                    f"问：{q}\n答：{a}" for q, a in self.chat_history
                )
# 5.  **只回答跟参考资料里有关的内容。如果参考资料里没有相关内容，切记不要自行回答问题！！！而是直接输出：“资料库中没有检索到相关内容”**
            prompt = f"""作为教学助手，你的回答应当帮助学习者深入理解。不要有任何开场白或过渡语，只输出正文。
            注重知识点的扩展和联系，保证回答的准确性和完整性。
            **为了优化Markdown结构，请注意以下几点以提升文档质量：**

1.  **提升可读性 (Readability)：** Markdown结构应该易于快速浏览和理解。请注意段落之间的空行，列表的缩进，以及避免过度使用粗体等影响阅读体验的元素。

2.  **构建清晰的层级结构 (Hierarchy)：** 使用合适的标题层级 (H2, H3, H4...) 来组织内容，确保内容逻辑清晰，层级分明。主标题用 H2, 子标题用 H3，以此类推。大标题也可以用“数字. [标题]”来区分结构。 使用列表 (有序列表和无序列表) 来组织并列的内容，例如步骤、要点、原因、例子等。

3.  **保持结构简洁 (Conciseness)：** 在保证信息完整性的前提下，尽量使用简洁的Markdown结构，避免不必要的复杂性。适度使用粗体、斜体等强调，但避免过度使用。

4.  **正确使用Markdown语法 (Correct Syntax)：** 确保Markdown语法正确无误，例如表格、链接、代码块等语法的正确使用。避免生成冗长的表格。
5.  **只回答跟参考资料里有关的内容。如果参考资料里没有相关内容，切记不要自行回答问题！！！而是直接输出：“资料库中没有检索到相关内容”**

            回答要深入而广泛，全面又完整。  
                另外要求：
                1. 必要时准确引用参考资料内容,只需要在小标题处标注参考资料来源（P+页数）即可，正文不需要标注。
                2. 逻辑清晰，层次分明
                3. 重点突出，联系紧密
                4. 便于理解和记忆
                5. 有助于构建知识体系
在回答的结尾，可以提出1-2个有助于构建知识体系的问题（小标题为“思考题”），并给出答案，引导学生进一步思考。


基于以下参考资料和历史对话回答问题:
参考资料:
{context}

历史对话:
{history_text}

问题：{query}"""

            # 生成响应
            response = self.model.generate_content(prompt)
            self.chat_history.append((query, response.text))
            if len(self.chat_history) > self.max_history:
                self.chat_history = self.chat_history[-self.max_history:]
            
            yield response.text
            
        except Exception as e:
            yield f"错误: {str(e)}"
    
    def _evaluate_doc_quality(self, content: str, query: str) -> float:
        """评估文档质量"""
        indicators = {
            'term_count': len(re.findall(r'[A-Z][a-z]+(?:[ -][A-Z][a-z]+)*', content)),
            'structure_score': 1 if re.search(r'。|；|！|？', content) else 0.5,
            'relevance_score': sum(q in content for q in jieba.cut(query)) / len(query)
        }
        
        weights = {'term_count': 0.3, 'structure_score': 0.3, 'relevance_score': 0.4}
        final_score = sum(score * weights[metric] for metric, score in indicators.items())
        
        return min(1.0, final_score)
    
    def update_knowledge_base(self, pdf_dir):
        """更新知识库"""
        from utils.pdf_loader import load_pdf, get_pdf_files
        from pymilvus import utility, Collection

        current_files = get_pdf_files(pdf_dir)
        if not current_files:
            print(f"警告: {pdf_dir} 目录下没有找到PDF文件")
            return

        processed_files = set()
        collection_name = "medical_knowledge_base"  # 假设集合名称是固定的
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
            for entity in collection.query(expr='id >= 0', output_fields=['source']):
                processed_files.add(entity['source'])

        new_files = current_files - processed_files

        if new_files:
            print("检测到新增PDF文件，进行增量更新...")
            for file in new_files:
                pdf_path = os.path.join(pdf_dir, file)
                load_pdf(pdf_path, self.vectorstore, self.embeddings)
            print("知识库已更新")
        else:
            print("未检测到新增PDF文件")
