import os
import sys
import json
import torch
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class RAGGenerator:
    """Class for generating responses using LLMs with retrieved context"""
    
    def __init__(self, 
                model_name_or_path: str,
                use_quantization: bool = True,
                device: str = "auto"):
        """
        Initialize the RAG Generator
        
        Args:
            model_name_or_path (str): HuggingFace model name or local path
            use_quantization (bool): Whether to use quantization for efficiency
            device (str): Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model_name_or_path = model_name_or_path
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            print(f"Initialized tokenizer for {model_name_or_path}")
        except Exception as e:
            print(f"Error initializing tokenizer: {e}")
            self.tokenizer = None
            
        # Initialize model with a fallback approach
        self.model = None
        self.pipeline = None
        self._initialize_model(use_quantization)
            
    def _initialize_model(self, use_quantization: bool):
        """
        Initialize the model with fallback methods
        
        Args:
            use_quantization (bool): Whether to use quantization
        """
        try:
            # Try loading with quantization if requested and on CUDA
            if use_quantization and self.device == "cuda":
                try:
                    # Try importing BitsAndBytesConfig
                    from transformers import BitsAndBytesConfig
                    
                    print(f"Loading {self.model_name_or_path} with 4-bit quantization...")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name_or_path,
                        device_map="auto",
                        quantization_config=quantization_config,
                    )
                    print("Model loaded with quantization")
                except Exception as e:
                    print(f"Quantization failed: {e}")
                    print("Falling back to standard loading...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name_or_path,
                        device_map="auto"
                    )
            else:
                # Standard loading
                print(f"Loading {self.model_name_or_path} on {self.device}...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    device_map=self.device
                )
                
            # Create generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            print(f"Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.pipeline = None
            
    def is_initialized(self) -> bool:
        """Check if the model is properly initialized"""
        return self.model is not None and self.tokenizer is not None
    
    def format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Format context chunks into a string for the prompt
        
        Args:
            context_chunks (list): List of context chunks with text and metadata
            
        Returns:
            str: Formatted context string
        """
        context_str = ""
        
        # Group chunks by source document
        chunks_by_source = {}
        for chunk in context_chunks:
            source = chunk.get('metadata', {}).get('paper_id', 'unknown')
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)
            
        # Format each source's chunks
        for source, chunks in chunks_by_source.items():
            if context_str:
                context_str += "\\n\\n"
                
            # Include source information
            context_str += f"Source: {source}\\n"
            
            # Add chunks with section information
            for chunk in chunks:
                text = chunk.get('text', '')
                metadata = chunk.get('metadata', {})
                section = metadata.get('section', 'unknown')
                
                context_str += f"[{section.upper()}] {text}\\n"
                
        return context_str
    
    def generate_prompt(self, query: str, context: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a complete prompt with system, context, and query
        
        Args:
            query (str): User query
            context (str): Retrieved context
            system_prompt (str, optional): Custom system prompt
            
        Returns:
            str: Complete prompt for the model
        """
        if system_prompt is None:
            system_prompt = """You are a helpful academic research assistant. \
Your goal is to answer questions accurately using only the context provided. \
If the information is not in the context, say that you don't know. \
Always cite the source of your information by mentioning the paper ID."""
            
        prompt = f"""### System:
{system_prompt}

### Context:
{context}

### User Question:
{query}

### Assistant:"""
        
        return prompt
    
    def generate_response(self, 
                         prompt: str, 
                         max_new_tokens: int = 512,
                         temperature: float = 0.7) -> str:
        """
        Generate a response from the model
        
        Args:
            prompt (str): Full prompt
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated response
        """
        if not self.is_initialized():
            return "Error: Model not properly initialized"
            
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                return_full_text=False
            )
            
            response = outputs[0]['generated_text']
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
            
    def answer_from_context(self,
                           query: str,
                           context_chunks: List[Dict[str, Any]],
                           system_prompt: Optional[str] = None,
                           max_new_tokens: int = 512,
                           temperature: float = 0.7) -> str:
        """
        Answer a query using retrieved context - main method for RAG
        
        Args:
            query (str): User query
            context_chunks (list): Retrieved context chunks
            system_prompt (str, optional): Custom system prompt
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated answer
        """
        # Check if model is initialized
        if not self.is_initialized():
            return "Error: Model not properly initialized"
            
        # Format context
        context = self.format_context(context_chunks)
        
        # Generate prompt
        prompt = self.generate_prompt(query, context, system_prompt)
        
        # Generate response
        response = self.generate_response(prompt, max_new_tokens, temperature)
        
        return response
        
    def save_config(self, config_path: str) -> bool:
        """
        Save the generator configuration
        
        Args:
            config_path (str): Path to save config
            
        Returns:
            bool: Success status
        """
        config = {
            "model_name_or_path": self.model_name_or_path,
            "device": self.device
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False