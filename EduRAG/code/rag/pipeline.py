import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

class EduRAGPipeline:
    """Complete RAG pipeline for academic papers"""
    
    def __init__(self, 
                 arxiv_api,
                 pdf_converter, 
                 text_chunker, 
                 retriever,
                 generator):
        """
        Initialize the EduRAG pipeline
        
        Args:
            arxiv_api: ArxivAPI instance
            pdf_converter: PDFConverter instance
            text_chunker: TextChunker instance
            retriever: RAGRetriever instance
            generator: RAGGenerator instance
        """
        self.arxiv_api = arxiv_api
        self.pdf_converter = pdf_converter
        self.text_chunker = text_chunker
        self.retriever = retriever
        self.generator = generator
        
        # Configuration
        self.collection_name = "edurag_collection"
        self.system_prompt = """You are EduRAG, an intelligent academic research assistant. 
Your goal is to provide accurate, helpful answers based on the academic papers in your knowledge base.
Use the provided context to answer the question. If the answer cannot be found in the context, acknowledge that.
Always cite your sources when providing information by mentioning the paper ID."""
    
    def search_and_process_papers(self, query: str, max_results: int = 5) -> List[str]:
        """
        Search for papers and process them through the pipeline
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of papers to retrieve
            
        Returns:
            list: List of processed paper IDs
        """
        # 1. Search and download papers
        print(f"Searching for papers on: {query}")
        papers = self.arxiv_api.search_papers(query, max_results=max_results)
        
        if not papers:
            print("No papers found for the query")
            return []
            
        print(f"Found {len(papers)} papers")
        
        # 2. Process papers
        processed_papers = []
        
        for paper in papers:
            try:
                # Download PDF
                pdf_path = self.arxiv_api.download_paper(paper)
                if not pdf_path:
                    print(f"Failed to download paper {paper.get_short_id()}")
                    continue
                    
                # Extract paper ID
                paper_id = paper.get_short_id()
                    
                # Convert PDF to structured text
                print(f"Processing paper: {paper_id}")
                structure, text_path = self.pdf_converter.process_pdf(pdf_path)
                
                if not text_path:
                    print(f"Failed to process paper {paper_id}")
                    continue
                    
                # Create chunks
                chunks, chunks_path = self.text_chunker.process_paper(text_path)
                
                if not chunks:
                    print(f"Failed to create chunks for paper {paper_id}")
                    continue
                    
                # Add to processed papers
                processed_papers.append(paper_id)
                print(f"Successfully processed paper: {paper_id}")
                
            except Exception as e:
                print(f"Error processing paper: {e}")
                
        return processed_papers
    
    def build_knowledge_base(self, chunks_dir: str) -> bool:
        """
        Build the knowledge base from processed chunks
        
        Args:
            chunks_dir (str): Directory containing chunk files
            
        Returns:
            bool: Success status
        """
        # Process all chunks into the vector database
        print(f"Building knowledge base from {chunks_dir}")
        success = self.retriever.process_directory(chunks_dir, self.collection_name)
        
        if success:
            print(f"Successfully built knowledge base")
        else:
            print(f"Failed to build knowledge base")
            
        return success
    
    def answer_question(self, 
                        question: str,
                        n_results: int = 5,
                        temperature: float = 0.7) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline
        
        Args:
            question (str): User question
            n_results (int): Number of chunks to retrieve
            temperature (float): Sampling temperature
            
        Returns:
            dict: Answer with metadata
        """
        # 1. Retrieve relevant chunks
        print(f"Retrieving context for question: {question}")
        context_chunks = self.retriever.query_collection(
            question, 
            self.collection_name,
            n_results=n_results
        )
        
        if not context_chunks:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "context_chunks": [],
                "sources": []
            }
            
        # 2. Generate answer
        print(f"Generating answer with {len(context_chunks)} context chunks")
        answer = self.generator.answer_from_context(
            question,
            context_chunks,
            system_prompt=self.system_prompt,
            temperature=temperature
        )
        
        # 3. Extract source information
        sources = []
        for chunk in context_chunks:
            metadata = chunk.get('metadata', {})
            source = {
                'paper_id': metadata.get('paper_id', 'unknown'),
                'section': metadata.get('section', 'unknown'),
                'score': chunk.get('score', 0.0)
            }
            if source not in sources:
                sources.append(source)
        
        # 4. Return complete response
        return {
            "answer": answer,
            "context_chunks": context_chunks,
            "sources": sources
        }
    
    def evaluate_models(self,
                        questions: List[str],
                        models: List[Dict[str, Any]],
                        n_results: int = 5) -> Dict[str, Any]:
        """
        Evaluate different models on the same questions
        
        Args:
            questions (list): List of questions to evaluate
            models (list): List of model configurations
            n_results (int): Number of context chunks to retrieve
            
        Returns:
            dict: Evaluation results
        """
        results = {
            "questions": questions,
            "models": [],
            "answers": {}
        }
        
        # Process each model
        for model_config in models:
            model_name = model_config.get('name', 'unknown')
            model_path = model_config.get('path')
            
            if not model_path:
                print(f"Missing model path for {model_name}")
                continue
                
            print(f"Evaluating model: {model_name}")
            
            # Initialize the generator with this model
            try:
                from rag.generator import RAGGenerator
                generator = RAGGenerator(
                    model_name_or_path=model_path,
                    use_quantization=model_config.get('quantization', True)
                )
                
                # Save original generator
                original_generator = self.generator
                # Set current generator to the new one
                self.generator = generator
                
                # Process each question
                model_results = []
                
                for question in questions:
                    print(f"Processing question: {question}")
                    
                    # Get answer
                    result = self.answer_question(
                        question,
                        n_results=n_results,
                        temperature=model_config.get('temperature', 0.7)
                    )
                    
                    model_results.append({
                        "question": question,
                        "answer": result.get('answer'),
                        "sources": result.get('sources')
                    })
                
                # Add to results
                results['answers'][model_name] = model_results
                results['models'].append({
                    "name": model_name,
                    "path": model_path
                })
                
                # Restore original generator
                self.generator = original_generator
                
            except Exception as e:
                print(f"Error evaluating model {model_name}: {e}")
        
        return results
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str) -> bool:
        """
        Save evaluation results to a JSON file
        
        Args:
            results (dict): Evaluation results
            output_path (str): Path to save results
            
        Returns:
            bool: Success status
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving evaluation results: {e}")
            return False