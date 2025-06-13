from manager import LLMManager
from vector.vector_store import VectorStore
from llm.llm_initiator import HuggingFaceProvider
from llm.model_locator import LocalModelLocation

async def main():
    print("Starting..")
    # Initialize providers
    API_token='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'
    llm_provider = HuggingFaceProvider(model_name="mistralai/Mistral-7B-Instruct-v0.3",api_token=API_token,model_location=LocalModelLocation(cache_dir="./model_cache"))
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Initialize RAG manager
    rag_manager = LLMManager(
        llm_provider=llm_provider,
        vector_store=vector_store
    )
    rag_manager.initialize()
    print("Initialization is done")
    # # Initialize document processor
    # processor = DocumentProcessor(
    #     chunk_size=1000,
    #     chunk_overlap=200
    # )
    
    # # Add documents
    # file_paths = ["document1.pdf", "document2.txt"]
    # rag_manager.process_and_add_documents(file_paths, processor)
    
    # Get response
    response = await rag_manager.get_response(
        "What are the two book mentioned here?",
        return_source_documents=True
    )
    
    print("Answer:", response["answer"])
    print("Sources:", response["sources"])


if __name__=="__main__":
    import asyncio
    asyncio.run(main())