# import PDFLoader
# import PineconeStore
# import EmbeddingGenerator
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional, Union, Any
# if __name__ == '__main__':
#     loader = PDFLoader('book_1.pdf')
#     generator = EmbeddingGenerator()
#     vector_store = PineconeStore()

#     text = loader.extract_text()

#     chunks, embeddings = generator.process_text(text, chunk_size=800)
    
    
#     vector_store.save_vectors(embeddings, {"id": "book_1", "source": "The_intelligent_investor_book.pdf"}, chunks)
 
#     vector_store.query("How to pick the best stock?")

class VectorStore:
    """Manages vector store operations for RAG"""
    def __init__(self):
        self.vectorstore = None
    
    # def __init__(self, 
    #              index_name: str,
    #              embedding_provider: Any,
    #              pinecone_api_key: str,
    #              pinecone_env: str):
    #     self.index_name = index_name
    #     self.embedding_provider = embedding_provider
    #     self.pinecone_api_key = pinecone_api_key
    #     self.pinecone_env = pinecone_env
    #     self.vectorstore = None
        
    def initialize_vectorstore(self):
        """Initialize Pinecone vector store"""
        # pinecone.init(
        #     api_key=self.pinecone_api_key,
        #     environment=self.pinecone_env
        # )
        
        # self.vectorstore = Pinecone.from_existing_index(
        #     index_name=self.index_name,
        #     embedding=self.embedding_provider.get_embeddings()
        # )
        pinecone_api_key = 'pcsk_2n7saV_MtrzwnaU2ycajidpd3jM8KdCDLgCB1sYRDcZYah697P2e5AZ6e21AjVGHjmZpyb'
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(host="https://stock-ai-assitant-4dniii9.svc.aped-4627-b74a.pinecone.io")
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2')
        self.vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
        
    def add_documents(self, documents: List[Any]):
        """Add documents to vector store"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
            
        self.vectorstore.add_documents(documents)
        
    def similarity_search(self, query: str, k: int = 4) -> List[Any]:
        """Perform similarity search"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
            
        return self.vectorstore.similarity_search(query, k=k)