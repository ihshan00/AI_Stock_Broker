import logging
from utils.config_loader import load_config,load_text_file
from llm.llm_initiator import initialize_llm
from vector.vector_store import VectorStore
from manager import LLMManager
import os
from langchain.prompts import PromptTemplate
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  
config_path = os.path.join(base_dir,"ai_project", "config", "config.yaml")
config = load_config(config_path)
print("Config: ",config)
# Initialize LLM
llm = initialize_llm(
    model_name=config.get("model_name"),
    api_token=config.get("api_key"),
    temperature=config.get("temperature"),
    cache_dir=config.get("cache_dir")
)

# Initialize VectorStore
vector_store = VectorStore()

# Initialize RAG manager
rag_manager = LLMManager(
    llm_provider=llm,
    vector_store=vector_store
)
rag_manager.initialize()
logging.info("Initialization is done")

# Load prompt template
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  
prompt_path = os.path.join(base_dir,config.get("prompt_template_path"))
prompt_template = load_text_file(prompt_path)

if not prompt_template:
    logging.error("Failed to load prompt template.")
    # Handle this error appropriately, maybe exit or use a default template

async def generate_text(prompt: str) -> str:
    logging.info(f"Generating text for prompt: {prompt}")
    # Assuming rag_manager.get_response can handle a PromptTemplate object
    # or you need to format the prompt manually
    formatted_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"]).format(context="...", question=prompt) 

    start_time = time.time()
    response = await rag_manager.get_response(
        formatted_prompt)
    end_time = time.time()

    print(f"Generation time: {end_time - start_time:.4f} seconds")
    return response["answer"]


async def update_vector_db(prompt: str) -> str:
    pass
