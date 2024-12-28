import PDFLoader
import PineconeStore
import EmbeddingGenerator


if __name__ == '__main__':
    loader = PDFLoader('book_1.pdf')
    generator = EmbeddingGenerator()
    vector_store = PineconeStore()

    text = loader.extract_text()

    chunks, embeddings = generator.process_text(text, chunk_size=800)
    
    
    vector_store.save_vectors(embeddings, {"id": "book_1", "source": "The_intelligent_investor_book.pdf"}, chunks)
 
    vector_store.query("How to pick the best stock?")