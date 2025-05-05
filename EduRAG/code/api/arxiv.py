import arxiv
import os
import time
from pathlib import Path

class ArxivAPI:
    """Class to interact with the ArXiv API and download papers"""
    
    def __init__(self, save_dir):
        """
        Initialize the ArXiv API handler
        
        Args:
            save_dir (str): Directory to save downloaded papers
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def search_papers(self, query, max_results=10, categories=None):
        """
        Search ArXiv for papers matching the query
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            categories (list): List of categories to filter by (e.g., ['cs.AI', 'cs.CL'])
            
        Returns:
            list: List of paper results
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        
        if categories:
            search = arxiv.Search(
                query=query + ' AND (' + ' OR '.join(f'cat:{c}' for c in categories) + ')',
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            
        results = list(search.results())
        return results
    
    def download_paper(self, paper, save_path=None):
        """
        Download a paper's PDF
        
        Args:
            paper (arxiv.Result): Paper result from search
            save_path (str, optional): Custom save path. If None, use the paper's ID
            
        Returns:
            str: Path to the downloaded PDF
        """
        if save_path is None:
            save_path = self.save_dir / f"{paper.get_short_id()}.pdf"
        else:
            save_path = Path(save_path)
            
        # Check if already downloaded
        if save_path.exists():
            print(f"Paper already exists at {save_path}")
            return str(save_path)
            
        # Download with rate limiting
        try:
            paper.download_pdf(filename=str(save_path))
            print(f"Downloaded paper to {save_path}")
            time.sleep(1)  # Rate limiting
            return str(save_path)
        except Exception as e:
            print(f"Error downloading paper: {e}")
            return None
    
    def get_paper_metadata(self, paper):
        """
        Extract metadata from a paper
        
        Args:
            paper (arxiv.Result): Paper result
            
        Returns:
            dict: Paper metadata
        """
        metadata = {
            'id': paper.get_short_id(),
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'abstract': paper.summary,
            'categories': paper.categories,
            'published': paper.published.strftime('%Y-%m-%d'),
            'pdf_url': paper.pdf_url,
            'entry_id': paper.entry_id,
        }
        return metadata
    
    def search_and_download(self, query, max_results=5, categories=None):
        """
        Search for papers and download them
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results
            categories (list): List of categories to filter by
            
        Returns:
            list: List of metadata dictionaries with download paths
        """
        papers = self.search_papers(query, max_results, categories)
        results = []
        
        for paper in papers:
            metadata = self.get_paper_metadata(paper)
            pdf_path = self.download_paper(paper)
            if pdf_path:
                metadata['local_pdf'] = pdf_path
            results.append(metadata)
        
        return results