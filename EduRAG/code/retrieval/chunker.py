import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import nltk
from nltk.tokenize import sent_tokenize

# Ensure we have the necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class TextChunker:
    """Class for chunking academic paper text into meaningful segments"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the text chunker
        
        Args:
            output_dir (str, optional): Directory to save chunked files
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
    
    def load_processed_paper(self, json_path: str) -> Dict[str, Any]:
        """
        Load a processed paper JSON file
        
        Args:
            json_path (str): Path to the processed paper JSON
            
        Returns:
            dict: Paper content
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading processed paper: {e}")
            return {}
    
    def split_text_by_sentences(self, text: str, max_sentences: int = 5) -> List[str]:
        """
        Split text into chunks of sentences
        
        Args:
            text (str): Text to split
            max_sentences (int): Maximum number of sentences per chunk
            
        Returns:
            list: List of text chunks
        """
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Create chunks of max_sentences
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk = ' '.join(sentences[i:i + max_sentences])
            chunks.append(chunk)
            
        return chunks
    
    def split_text_by_tokens(self, text: str, max_tokens: int = 256) -> List[str]:
        """
        Split text into chunks based on approximate token count
        
        Args:
            text (str): Text to split
            max_tokens (int): Maximum tokens per chunk (approximate)
            
        Returns:
            list: List of text chunks
        """
        # Simple word-based tokenization (an approximation)
        words = text.split()
        
        # Create chunks of approximately max_tokens
        chunks = []
        chunk = []
        token_count = 0
        
        for word in words:
            # Approximate token count (may vary based on tokenizer)
            word_tokens = max(1, len(word) // 4)
            
            if token_count + word_tokens > max_tokens and chunk:
                chunks.append(' '.join(chunk))
                chunk = []
                token_count = 0
                
            chunk.append(word)
            token_count += word_tokens
            
        if chunk:
            chunks.append(' '.join(chunk))
            
        return chunks
    
    def create_semantic_chunks(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create semantically meaningful chunks from paper sections
        
        Args:
            paper (dict): Processed paper content
            
        Returns:
            list: List of chunk dictionaries
        """
        chunks = []
        paper_id = paper.get('filename', 'unknown')
        source_pdf = paper.get('source_pdf', '')
        
        # Function to add a chunk with metadata
        def add_chunk(text, section, index):
            if not text.strip():
                return
                
            chunks.append({
                'paper_id': paper_id,
                'source': source_pdf,
                'section': section,
                'chunk_id': f"{paper_id}_{section}_{index}",
                'text': text.strip(),
                'tokens': len(text.split())  # Approximate token count
            })
        
        # Process each section with appropriate chunking strategy
        for section in ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']:
            content = paper.get(section, '')
            if not content:
                continue
                
            # Use different chunking strategies based on section
            if section == 'abstract':
                # Keep abstract as a single chunk unless it's very long
                if len(content.split()) > 300:  # If abstract is unusually long
                    section_chunks = self.split_text_by_tokens(content, 200)
                else:
                    section_chunks = [content]
            else:
                # Use token-based chunking for other sections
                section_chunks = self.split_text_by_tokens(content, 256)
                
            # Add chunks with metadata
            for i, chunk_text in enumerate(section_chunks):
                add_chunk(chunk_text, section, i)
        
        # Add title as a special chunk
        title = paper.get('title', '')
        if title:
            add_chunk(title, 'title', 0)
            
        # Add keywords as a special chunk if present
        keywords = paper.get('keywords', '')
        if keywords:
            add_chunk(keywords, 'keywords', 0)
            
        return chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], paper_id: str) -> str:
        """
        Save chunks to a JSON file
        
        Args:
            chunks (list): List of chunk dictionaries
            paper_id (str): Paper identifier
            
        Returns:
            str: Path to the saved file
        """
        if not self.output_dir:
            return None
            
        output_path = self.output_dir / f"chunks_{paper_id}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
            
        return str(output_path)
    
    def process_paper(self, json_path: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Process a paper to create chunks
        
        Args:
            json_path (str): Path to processed paper JSON
            
        Returns:
            tuple: (list of chunks, save path or None)
        """
        # Load the processed paper
        paper = self.load_processed_paper(json_path)
        if not paper:
            return [], None
            
        # Extract paper ID
        paper_id = paper.get('filename', Path(json_path).stem)
        
        # Create chunks
        chunks = self.create_semantic_chunks(paper)
        
        # Save chunks if output directory is specified
        save_path = None
        if self.output_dir:
            save_path = self.save_chunks(chunks, paper_id)
            
        return chunks, save_path
    
    def batch_process_papers(self, directory: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process multiple papers in a directory
        
        Args:
            directory (str): Directory containing processed paper JSONs
            
        Returns:
            dict: Dictionary mapping paper IDs to chunks
        """
        directory_path = Path(directory)
        results = {}
        
        for json_file in directory_path.glob('*.json'):
            try:
                chunks, _ = self.process_paper(str(json_file))
                paper_id = json_file.stem.replace('processed_', '')
                results[paper_id] = chunks
                print(f"Processed {json_file.name}: {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
                
        return results