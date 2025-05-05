import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    """Class for embedding chunks and retrieving relevant context for RAG"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: Optional[str] = None):
        """
        Initialize the RAG retriever
        
        Args:
            embedding_model (str): Name of the sentence-transformers model to use
            persist_directory (str, optional): Directory to persist ChromaDB
        """
        self.embedding_model_name = embedding_model
        
        # Initialize the embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            print(f"Initialized embedding model: {embedding_model}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.embedding_model = None
            
        # Initialize ChromaDB
        self.persist_directory = persist_directory
        if persist_directory:
            self.chroma_client = chromadb.PersistentClient(persist_directory)
        else:
            self.chroma_client = chromadb.Client()
            
        # Use sentence-transformers embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
    
    def create_collection(self, collection_name: str) -> Any:
        """
        Create a ChromaDB collection
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            Collection: ChromaDB collection
        """
        try:
            # Check if collection exists and delete if it does
            try:
                self.chroma_client.delete_collection(collection_name)
                print(f"Deleted existing collection: {collection_name}")
            except:
                pass
                
            # Create new collection
            collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Created collection: {collection_name}")
            return collection
        except Exception as e:
            print(f"Error creating collection: {e}")
            return None
    
    def get_collection(self, collection_name: str) -> Any:
        """
        Get an existing ChromaDB collection
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            Collection: ChromaDB collection or None if not found
        """
        try:
            collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            return collection
        except Exception as e:
            print(f"Collection not found: {collection_name}")
            return None
    
    def add_chunks_to_collection(self, 
                                chunks: List[Dict[str, Any]], 
                                collection_name: str) -> bool:
        """
        Add document chunks to a ChromaDB collection
        
        Args:
            chunks (list): List of chunk dictionaries
            collection_name (str): Name of the collection
            
        Returns:
            bool: Success status
        """
        # Get or create collection
        collection = self.get_collection(collection_name)
        if not collection:
            collection = self.create_collection(collection_name)
            if not collection:
                return False
                
        # Prepare data for insertion
        chunk_ids = []
        chunk_texts = []
        chunk_metadata = []
        
        for chunk in chunks:
            # Create a unique ID for each chunk
            chunk_id = chunk.get('chunk_id', f"chunk_{len(chunk_ids)}")
            
            # Get the text content
            text = chunk.get('text', '')
            if not text.strip():
                continue
                
            # Prepare metadata (everything except the text)
            metadata = {k: v for k, v in chunk.items() if k != 'text'}
            
            chunk_ids.append(chunk_id)
            chunk_texts.append(text)
            chunk_metadata.append(metadata)
            
        # Add documents to collection
        if chunk_ids:
            try:
                collection.add(
                    ids=chunk_ids,
                    documents=chunk_texts,
                    metadatas=chunk_metadata
                )
                print(f"Added {len(chunk_ids)} chunks to collection {collection_name}")
                return True
            except Exception as e:
                print(f"Error adding chunks to collection: {e}")
                return False
        else:
            print("No valid chunks to add")
            return False
    
    def load_chunks_from_file(self, chunks_file: str) -> List[Dict[str, Any]]:
        """
        Load chunks from a JSON file
        
        Args:
            chunks_file (str): Path to chunks JSON file
            
        Returns:
            list: List of chunk dictionaries
        """
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            return chunks
        except Exception as e:
            print(f"Error loading chunks from file: {e}")
            return []
    
    def query_collection(self, 
                         query: str, 
                         collection_name: str,
                         n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query a collection for relevant chunks
        
        Args:
            query (str): Query string
            collection_name (str): Name of the collection
            n_results (int): Number of results to return
            
        Returns:
            list: List of relevant chunks with scores
        """
        # Get collection
        collection = self.get_collection(collection_name)
        if not collection:
            print(f"Collection not found: {collection_name}")
            return []
            
        # Query the collection
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0] if 'distances' in results else [0] * len(documents)
                
                for i in range(len(documents)):
                    result = {
                        'text': documents[i],
                        'metadata': metadatas[i],
                        'score': 1.0 - distances[i] if distances[i] <= 1.0 else 0.0  # Convert distance to similarity score
                    }
                    formatted_results.append(result)
                    
            return formatted_results
        except Exception as e:
            print(f"Error querying collection: {e}")
            return []
    
    def process_directory(self, chunks_dir: str, collection_name: str) -> bool:
        """
        Process all chunk files in a directory
        
        Args:
            chunks_dir (str): Directory containing chunk JSON files
            collection_name (str): Name of the collection to add chunks to
            
        Returns:
            bool: Success status
        """
        chunks_path = Path(chunks_dir)
        success = True
        
        # Create a new collection
        collection = self.create_collection(collection_name)
        if not collection:
            return False
            
        # Process each chunk file
        for chunks_file in chunks_path.glob('chunks_*.json'):
            try:
                print(f"Processing: {chunks_file.name}")
                chunks = self.load_chunks_from_file(str(chunks_file))
                if chunks:
                    if not self.add_chunks_to_collection(chunks, collection_name):
                        success = False
            except Exception as e:
                print(f"Error processing {chunks_file.name}: {e}")
                success = False
                
        return success