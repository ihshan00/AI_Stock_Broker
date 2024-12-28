import fitz 
from pinecone import Pinecone,ServerlessSpec

from dotenv import load_dotenv
import os
import tiktoken
from sentence_transformers import SentenceTransformer


class PineconeStore:
    def __init__(self, environment="us-east-1"):
        # Load API key from environment variables
        pinecone_api_key = "pcsk_4mkcnx_Eay9hXXKjz8DimVPPf2bG83G9uCFgS9XYtPtCnPW3JcaeaE1KwLVR8aeJdBDtuL"
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        # Create a Pinecone instance
        self.pc = Pinecone(api_key=pinecone_api_key)

        # Define the index name
        self.index_name = "stock-ai-assitant"

        # Check if the index exists, if not create it
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

    def save_vectors(self, vectors, metadata, chunks):
        # Get the index
        index = self.pc.Index(self.index_name)

        # Iterate over the embeddings and save each one with unique metadata
        for i, vector in enumerate(vectors):
            vector_id = f"{metadata['id']}_chunk_{i}"  # Unique ID for each chunk
            chunk_metadata = {
                "id": vector_id,
                "source": metadata["source"],
                "chunk": i,
                "text": chunks[i]  # Add the text of the chunk here
            }
            # Upsert each vector with its corresponding metadata
            index.upsert(vectors=[(vector_id, vector, chunk_metadata)])

            
    def query(sel,query):
      pc = Pinecone(api_key="pcsk_4mkcnx_Eay9hXXKjz8DimVPPf2bG83G9uCFgS9XYtPtCnPW3JcaeaE1KwLVR8aeJdBDtuL")

      # To get the unique host for an index, 
      index = pc.Index(host="https://stock-ai-assitant-4dniii9.svc.aped-4627-b74a.pinecone.io")

      query_text = query
      model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
      embedding = model.encode(query_text)
      embedding_list = embedding.tolist()  
      result=index.query(
          vector=embedding_list,
          top_k=3,
          include_values=False,
          include_metadata=True
      )

      for match in result["matches"]:
          print(f"ID: {match['id']}, Score: {match['score']}, Metadata: {match.get('metadata')['text']}")
          print(" ")



