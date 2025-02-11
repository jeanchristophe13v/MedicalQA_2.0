from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import os

load_dotenv()

class EmbeddingModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name="TencentBAC/Conan-embedding-v1",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder="./embeddings_cache"
        )

    def embed_query(self, query):
        return self.embeddings.embed_query(query)

embedding_model = EmbeddingModel()