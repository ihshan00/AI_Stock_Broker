import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from exec import generate_text, update_vector_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    prompt: str

class UpdateRequest(BaseModel):
    data: str

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        generated_text = await generate_text(request.prompt)
        return {"response": generated_text}
    except Exception as e:
        logging.error(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-vector-db")
async def update_endpoint(request: UpdateRequest):
    try:
        updated = await update_vector_db(request.data)
        return {"response": updated}
    except Exception as e:
        logging.error(f"Error in update endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
