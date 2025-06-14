from langchain.chains import LLMChain, RetrievalQA
from typing import List, Dict, Optional, Union, Any

class LLMManager:
    """Manager for RAG operations with LLM integration"""
    
    def __init__(
        self,
        llm_provider: Any,
        vector_store: Any,
        memory_size: int = 5
    ):
        self.llm_provider = llm_provider
        self.vector_store = vector_store
        # self.memory = ConversationBufferMemory(
        #     memory_key="chat_history",
        #     return_messages=True,
        #     k=memory_size
        # )
        self.qa_chain = None
        
        # Default RAG prompt template
        self.default_template = """
        You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Question: {question}
        
        Helpful Answer:"""
        
    def initialize(self):
        """Initialize the RAG system"""
        self.vector_store.initialize_vectorstore()
        
        # prompt = PromptTemplate(
        #     input_variables=["context", "chat_history", "question"],
        #     template=self.default_template
        # )
        retriever=self.vector_store.vectorstore.as_retriever()
        llm=self.llm_provider.get_llm()
        print("Line 44 Retriever",retriever)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            memory=None,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": None
            }
        )
        
    # def process_and_add_documents(self, 
    #                             file_paths: List[str], 
    #                             processor: DocumentProcessor):
    #     """Process and add documents to the vector store"""
    #     documents = processor.load_documents(file_paths)
    #     chunks = processor.process_documents(documents)
    #     self.vector_store.add_documents(chunks)
        
    async def get_response(self, 
                          question: str, 
                          return_source_documents: bool = False) -> Union[str, Dict]:
        """Get response using RAG"""
        if not self.qa_chain:
            raise ValueError("RAG system not initialized")
        
        result =  self.qa_chain.invoke(
            input=question
        )
        
        if result:
            # Assuming 'result' is a dictionary and contains the expected keys
            
            return {
                "answer": result.get("result"),  # Avoids KeyError if "result" is missing
                "sources": [doc.metadata for doc in result.get("source_documents", [])]
            }
        return 
