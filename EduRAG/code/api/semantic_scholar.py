import requests
import json
import time
import os
from pathlib import Path

class SemanticScholarAPI:
    """Class to interact with the Semantic Scholar API"""
    
    def __init__(self, save_dir=None, api_key=None):
        """
        Initialize the Semantic Scholar API handler
        
        Args:
            save_dir (str): Directory to save API responses
            api_key (str, optional): API key for Semantic Scholar
        """
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key
            
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
    
    def _handle_request(self, url, params=None):
        """
        Handle API requests with rate limiting and error handling
        
        Args:
            url (str): API endpoint URL
            params (dict, optional): Request parameters
            
        Returns:
            dict: API response
        """
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            time.sleep(1)  # Rate limiting
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return None
    
    def search_papers(self, query, limit=10, fields=None):
        """
        Search for papers by query
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            fields (list): Fields to include in the response
            
        Returns:
            dict: Search results
        """
        endpoint = f"{self.base_url}/paper/search"
        
        if fields is None:
            fields = ["paperId", "title", "abstract", "year", "authors", "venue", "url"]
            
        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(fields)
        }
        
        return self._handle_request(endpoint, params)
    
    def get_paper_details(self, paper_id, fields=None):
        """
        Get details for a specific paper
        
        Args:
            paper_id (str): Paper ID
            fields (list): Fields to include in the response
            
        Returns:
            dict: Paper details
        """
        endpoint = f"{self.base_url}/paper/{paper_id}"
        
        if fields is None:
            fields = [
                "paperId", "title", "abstract", "year", "authors", "venue",
                "referenceCount", "citationCount", "influentialCitationCount",
                "tldr", "url"
            ]
            
        params = {
            "fields": ",".join(fields)
        }
        
        return self._handle_request(endpoint, params)
    
    def get_paper_citations(self, paper_id, limit=10, fields=None):
        """
        Get citations for a paper
        
        Args:
            paper_id (str): Paper ID
            limit (int): Maximum number of citations to return
            fields (list): Fields to include in the response
            
        Returns:
            dict: Citation data
        """
        endpoint = f"{self.base_url}/paper/{paper_id}/citations"
        
        if fields is None:
            fields = ["paperId", "title", "year", "authors"]
            
        params = {
            "limit": limit,
            "fields": ",".join(fields)
        }
        
        return self._handle_request(endpoint, params)
    
    def get_paper_references(self, paper_id, limit=10, fields=None):
        """
        Get references for a paper
        
        Args:
            paper_id (str): Paper ID
            limit (int): Maximum number of references to return
            fields (list): Fields to include in the response
            
        Returns:
            dict: Reference data
        """
        endpoint = f"{self.base_url}/paper/{paper_id}/references"
        
        if fields is None:
            fields = ["paperId", "title", "year", "authors"]
            
        params = {
            "limit": limit,
            "fields": ",".join(fields)
        }
        
        return self._handle_request(endpoint, params)
    
    def find_paper_by_arxiv_id(self, arxiv_id, fields=None):
        """
        Find a paper by its ArXiv ID
        
        Args:
            arxiv_id (str): ArXiv ID (with or without the 'arXiv:' prefix)
            fields (list): Fields to include in the response
            
        Returns:
            dict: Paper details
        """
        # Remove 'arXiv:' prefix if present
        if arxiv_id.startswith('arXiv:'):
            arxiv_id = arxiv_id[6:]
            
        endpoint = f"{self.base_url}/paper/arXiv:{arxiv_id}"
        
        if fields is None:
            fields = ["paperId", "title", "abstract", "year", "authors", "venue", "url"]
            
        params = {
            "fields": ",".join(fields)
        }
        
        return self._handle_request(endpoint, params)
    
    def save_paper_data(self, paper_id, data, prefix="paper_"):
        """
        Save paper data to a JSON file
        
        Args:
            paper_id (str): Paper ID
            data (dict): Paper data to save
            prefix (str): Prefix for the filename
            
        Returns:
            str: Path to the saved file
        """
        if not self.save_dir:
            return None
            
        # Clean paper_id for filename
        clean_id = paper_id.replace('/', '_').replace('\\\\', '_')
        filename = f"{prefix}{clean_id}.json"
        file_path = self.save_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return str(file_path)