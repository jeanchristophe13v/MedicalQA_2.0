from dataclasses import dataclass
import re
from typing import List, Dict, Any
import jieba
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class TextStats:
    avg_sentence_length: float
    avg_paragraph_length: float
    term_density: float

class AdaptiveMedicalSplitter:
    def __init__(self):
        self.term_patterns = [
            r'[A-Z][a-z]+(?:[ -][A-Z][a-z]+)*',  # 医学术语
            r'\d+(?:\.\d+)?%?(?:[ -][A-Za-z]+)+', # 剂量/数值
            r'[A-Z]{2,}',  # 缩写
        ]

        self.section_patterns = {
            'chapter': r'^第[一二三四五六七八九十百]+章|^第\d+章',
            'section': r'^\d+\.\d+\s+[^\n]+|^第[一二三四五六七八九十百]+节|^第\d+节',
            'subsection': r'^\d+\.\d+\.\d+\s+[^\n]+',
            'item':r'^[一二三四五六七八九十]、|\d+、|\d+\. |\（\d+\）|\d+\)'
        }
        # 添加医学词典
        medical_terms = [
            "药理学", "病理学", "生理学", "解剖学", "生物化学",
            "免疫学", "微生物学", "寄生虫学", "内科学", "外科学",
            "妇产科学", "儿科学", "神经病学", "精神病学", "肿瘤学",
            "心脏病学", "肾脏病学", "呼吸病学", "消化病学", "内分泌学",
            "血液学", "传染病学", "皮肤病学", "眼科学", "耳鼻喉科学",
            "口腔科学", "麻醉学", "急诊医学", "康复医学", "医学影像学",
            "临床药理学", "药物代谢动力学", "药物效应动力学", "不良反应", "药物相互作用",
            "剂量", "给药途径", "适应症", "禁忌症", "注意事项",
            "处方", "非处方药", "中药", "西药", "生物制剂",
            "疫苗", "诊断", "治疗", "预防", "预后",
            "症状", "体征", "实验室检查", "影像学检查", "病理学检查",
            "基因检测", "临床试验", "循证医学", "安慰剂", "双盲",
            "随机对照试验", "Meta分析", "系统评价", "指南", "共识",
            "发生机制", "病因", "临床表现", "伴随症状", "参考区间",
            "药理作用及机制", "药理作用", "临床应用", "不良反应及注意事项",
            "体内过程", "病因和病理", "临床表现和并发症", "诊断与鉴别诊断",
            "解剖概要", "应用解剖", "病理生理", "分型", "病理", "生理", "分类"
        ]
        for term in medical_terms:
            jieba.add_word(term)

    def analyze_text(self, text: str) -> TextStats:
        """分析文本特征"""
        sentences = re.split(r'[。！？]', text)
        paragraphs = text.split('\n\n')

        avg_sent_len = sum(len(s) for s in sentences) / max(len(sentences), 1)
        avg_para_len = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)

        term_count = sum(len(re.findall(pattern, text)) for pattern in self.term_patterns)
        term_density = term_count / len(text) if text else 0

        return TextStats(avg_sent_len, avg_para_len, term_density)

    def get_optimal_chunk_size(self, stats: TextStats) -> Dict[str, int]:
        """根据文本统计信息获取最佳块大小"""
        # 调整 chunk_size 和 chunk_overlap
        if stats.term_density > 0.05:
            chunk_size = 700  # 调整为 700 个词
            chunk_overlap = 200 # 调整为 200 个词
        else:
            chunk_size = 500  # 调整为 500 个词
            chunk_overlap = 150  # 调整为 150 个词

        return {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }

    def split_document(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        # 预处理：去除不必要的空格和换行符
        text = re.sub(r'\s+', ' ', text).strip()

        stats = self.analyze_text(text)
        params = self.get_optimal_chunk_size(stats)

        # chunk_size 和 chunk_overlap 使用词数而不是字符数
        chunk_size_words = params["chunk_size"]
        chunk_overlap_words = params["chunk_overlap"]
        
        documents = []

        # 多级标题分割
        def split_by_pattern(text, pattern_name):
            pattern = self.section_patterns[pattern_name]
            parts = re.split(pattern, text)
            titles = re.findall(pattern, text)
            return parts, titles
        
        chapters, chapter_titles = split_by_pattern(text, 'chapter')
        #如果无法匹配到章节
        if len(chapters) <= 1:
            chapters = [text]
            chapter_titles = ['']

        chapter_index = 0
        for i, chapter_content in enumerate(chapters):
            if not chapter_content.strip():
                continue

            chapter_title = chapter_titles[chapter_index] if chapter_index < len(chapter_titles) else ''
            chapter_index += 1

            sections, section_titles = split_by_pattern(chapter_content, 'section')
            section_index = 0
            
            for j, section_content in enumerate(sections):
                if not section_content.strip():
                    continue

                section_title = section_titles[section_index] if section_index < len(section_titles) else ''
                section_index += 1
                
                items, item_titles = split_by_pattern(section_content, 'item')
                item_index = 0

                for k, item_content in enumerate(items):
                    if not item_content.strip():
                        continue
                    
                    item_title = item_titles[item_index] if item_index < len(item_titles) else ''
                    item_index += 1

                    # 使用 jieba 分词
                    words = list(jieba.cut(item_content))

                    chunks = []
                    current_chunk = []
                    current_length = 0

                    for word in words:
                        current_chunk.append(word)
                        current_length += 1
                        if current_length >= chunk_size_words:
                            chunks.append("".join(current_chunk))
                            current_chunk = current_chunk[-chunk_overlap_words:]
                            current_length = len(current_chunk)

                    if current_chunk:
                        chunks.append("".join(current_chunk))

                    def clean_chunk(chunk: str) -> str:
                        if not chunk.endswith(("。", "！", "？")):
                            last_period = max(chunk.rfind("。"), chunk.rfind("！"), chunk.rfind("？"))
                            if last_period != -1:
                                chunk = chunk[:last_period + 1]
                        return chunk.strip()


                    for l, chunk in enumerate(chunks):
                        clean_text = clean_chunk(chunk)
                        if len(clean_text) > 4500:
                            clean_text = clean_text[:4500]
                        if not clean_text:
                            continue

                        # 修改元数据格式，只保留简单类型
                        chunk_metadata = {
                            **(metadata or {}),
                            "chapter_title": chapter_title,  # 添加章节标题
                            "section_title": section_title, # 添加节标题
                            "item_title": item_title, # 添加小标题
                            "chunk_index": str(l),
                            "chunk_total": str(len(chunks)),
                            "chunk_size": str(params["chunk_size"]),
                            "chunk_overlap": str(params["chunk_overlap"])
                        }

                        documents.append(Document(
                            page_content=clean_text,
                            metadata=chunk_metadata
                        ))
        #后处理，合并短chunk
        merged_documents = []
        temp_doc = None
        for doc in documents:
            if temp_doc is None:
                temp_doc = doc
            elif len(temp_doc.page_content) + len(doc.page_content) < params["chunk_size"] * 0.7:
                temp_doc.page_content += doc.page_content
                #只更新chunk_total
                temp_doc.metadata["chunk_total"] = str(int(temp_doc.metadata["chunk_total"]) + int(doc.metadata["chunk_total"]))

            else:
                merged_documents.append(temp_doc)
                temp_doc = doc
        if temp_doc:
            merged_documents.append(temp_doc)
        return merged_documents