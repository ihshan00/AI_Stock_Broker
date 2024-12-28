# Libraries
import fitz 
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv
import os
import tiktoken
from sentence_transformers import SentenceTransformer

class PDFLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
