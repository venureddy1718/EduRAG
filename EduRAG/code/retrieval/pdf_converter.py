import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Import PDF processing libraries
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class PDFConverter:
    """Class for converting academic PDF papers to structured text"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the PDF converter
        
        Args:
            output_dir (str, optional): Directory to save processed text files
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract plain text from a PDF file
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text
        """
        try:
            text = extract_text(pdf_path)
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text (str): Raw text extracted from PDF
            
        Returns:
            str: Cleaned text
        """
        # Replace multiple spaces with a single space
        text = re.sub(r'\\s+', ' ', text)
        
        # Remove form feed characters
        text = text.replace('\\f', ' ')
        
        # Remove unnecessary line breaks
        text = re.sub(r'(?<!\\.)\\n(?![A-Z])', ' ', text)
        
        # Fix hyphenated words
        text = re.sub(r'(\\w+)-\\s*\\n(\\w+)', r'\\1\\2', text)
        
        return text.strip()
    
    def extract_structured_content(self, pdf_path: str) -> Dict:
        """
        Extract structured content from a research paper PDF
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Structured content with sections
        """
        # Extract full text first
        full_text = self.extract_text_from_pdf(pdf_path)
        
        # Identify common section headers in academic papers
        section_patterns = [
            # Title patterns
            r'^(.*?)(?=Abstract|ABSTRACT|abstract)',
            # Abstract
            r'(?:Abstract|ABSTRACT)[:\\s]*(.*?)(?=(?:Introduction|INTRODUCTION|Keywords|KEYWORDS|\\d+\\.|\\n\\n\\d+\\s+\\w+))',
            # Keywords
            r'(?:Keywords|KEYWORDS)[:\\s]*(.*?)(?=(?:Introduction|INTRODUCTION|\\d+\\.|\\n\\n\\d+\\s+\\w+))',
            # Introduction
            r'(?:\\n|^)(?:\\d+\\s+)?(?:Introduction|INTRODUCTION)[\\s:.]*(.*?)(?=(?:\\n\\d+\\.|\\n\\d+\\s+\\w+|\\n\\n\\d+\\s+\\w+))',
            # Methods/Methodology
            r'(?:\\n|^)(?:\\d+\\s+)?(?:Methods|Method|Methodology|METHODS|METHOD)[\\s:.]*(.*?)(?=(?:\\n\\d+\\.|\\n\\d+\\s+\\w+|\\n\\n\\d+\\s+\\w+))',
            # Results
            r'(?:\\n|^)(?:\\d+\\s+)?(?:Results|RESULTS)[\\s:.]*(.*?)(?=(?:\\n\\d+\\.|\\n\\d+\\s+\\w+|\\n\\n\\d+\\s+\\w+))',
            # Discussion
            r'(?:\\n|^)(?:\\d+\\s+)?(?:Discussion|DISCUSSION)[\\s:.]*(.*?)(?=(?:\\n\\d+\\.|\\n\\d+\\s+\\w+|\\n\\n\\d+\\s+\\w+))',
            # Conclusion
            r'(?:\\n|^)(?:\\d+\\s+)?(?:Conclusion|Conclusions|CONCLUSION|CONCLUSIONS)[\\s:.]*(.*?)(?=(?:\\n\\d+\\.|\\n\\d+\\s+\\w+|\\n\\n\\d+\\s+\\w+|References|REFERENCES|Bibliography|BIBLIOGRAPHY))',
            # References
            r'(?:\\n|^)(?:\\d+\\s+)?(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)[\\s:.]*(.*?)(?=$)'
        ]
        
        # Initialize the structure
        structure = {
            'title': '',
            'abstract': '',
            'keywords': '',
            'introduction': '',
            'methods': '',
            'results': '',
            'discussion': '',
            'conclusion': '',
            'references': '',
            'full_text': full_text
        }
        
        # Extract each section using regex patterns
        for i, pattern in enumerate(section_patterns):
            matches = re.search(pattern, full_text, re.DOTALL)
            if matches:
                # Get the section name based on index
                section_names = ['title', 'abstract', 'keywords', 'introduction', 'methods', 'results', 'discussion', 'conclusion', 'references']
                if i < len(section_names):
                    section_content = matches.group(1).strip()
                    structure[section_names[i]] = self.clean_text(section_content)
        
        return structure
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text (str): Text to split
            
        Returns:
            list: List of sentences
        """
        sentences = sent_tokenize(text)
        return sentences
    
    def save_processed_text(self, pdf_path: str, structure: Dict, prefix: str = "processed_") -> str:
        """
        Save processed text to a JSON file
        
        Args:
            pdf_path (str): Original PDF path
            structure (dict): Structured content
            prefix (str): Prefix for the output filename
            
        Returns:
            str: Path to the saved file
        """
        if not self.output_dir:
            return None
            
        # Generate output filename based on the PDF filename
        pdf_name = Path(pdf_path).stem
        output_file = self.output_dir / f"{prefix}{pdf_name}.json"
        
        # Add additional metadata
        structure['source_pdf'] = pdf_path
        structure['filename'] = pdf_name
        
        # Write to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)
            
        return str(output_file)
    
    def process_pdf(self, pdf_path: str) -> Tuple[Dict, Optional[str]]:
        """
        Process a PDF file end-to-end
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            tuple: (structured content, save path or None)
        """
        # Extract structured content
        structure = self.extract_structured_content(pdf_path)
        
        # Save if output directory is specified
        save_path = None
        if self.output_dir:
            save_path = self.save_processed_text(pdf_path, structure)
            
        return structure, save_path