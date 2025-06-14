import logging
from src.utils.config_loader import load_config
from src.llm.llm_initiator import initialize_llm
from src.vector.vector_store import VectorStore
from manager import LLMManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
config = load_config("ai_project/config/config.yaml")

# Initialize LLM
llm = initialize_llm(
    model_name=config.get("model_name"),
    api_token=config.get("api_key"),
    temperature=config.get("temperature"),
    cache_dir=config.get("cache_dir")
)

# Initialize VectorStore
vector_store = VectorStore(config.get("vector_store_path"))

# Initialize RAG manager
rag_manager = LLMManager(
    llm_provider=llm,
    vector_store=vector_store
)
rag_manager.initialize()
logging.info("Initialization is done")

# Load prompt template
prompt_template = load_text_file(config.get("prompt_template_path"))
if not prompt_template:
    logging.error("Failed to load prompt template.")
    # Handle this error appropriately, maybe exit or use a default template

async def generate_text(prompt: str) -> str:
    logging.info(f"Generating text for prompt: {prompt}")
    # Assuming rag_manager.get_response can handle a PromptTemplate object
    # or you need to format the prompt manually
    formatted_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"]).format(context="...", question=prompt) # Need to get context
    response = await rag_manager.get_response(
        formatted_prompt,
        return_source_documents=True
    )
    return response["answer"]


async def update_vector_db(data: str) -> str:
    logging.info(f"Updating vector database with data: {data}")
    # Assuming 'data' is a file path or list of file paths
    # Need to integrate document processing and adding logic here
    # For now, returning a placeholder
    return "Vector Database Update Initiated"
