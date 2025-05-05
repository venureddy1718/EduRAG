# EduRAG: A Smart Learning Assistant for Academic and Research Papers

## Project Overview
EduRAG is a Retrieval Augmented Generation (RAG) system designed to assist students and researchers in comprehending complex academic papers. By leveraging state-of-the-art language models and advanced retrieval techniques, EduRAG simplifies technical content, provides concise summaries, and answers specific questions about research papers.

This project implements and compares three different language models (Phi-2, Mistral-7B, and LLaMA 3) to identify the optimal approach for academic content understanding.

## Problem Statement
Research papers are often too complex and time-consuming to read, making it challenging for students and researchers to extract valuable insights efficiently. This creates barriers to knowledge acquisition and slows down research progress.

## Solution
EduRAG addresses these challenges through:

- **Concise Summaries**: Simplifies complex papers into easy-to-understand summaries
- **Contextual Insights**: Clarifies important concepts and research results
- **Q&A Feature**: Allows users to ask specific questions and receive accurate answers based on paper content

## System Architecture
The EduRAG system consists of five main components:

1. **Data Ingestion**: Collects papers from ArXiv and Semantic Scholar APIs
2. **Document Processing**: Converts PDFs into structured text and splits them into meaningful chunks
3. **Vector Storage**: Creates and stores embeddings for efficient similarity search
4. **Retrieval**: Identifies the most relevant text chunks based on user queries
5. **Response Generation**: Leverages LLMs to produce coherent and contextually accurate answers

![EduRAG model comparison](https://github.com/user-attachments/assets/9578f95f-5858-4042-a6fe-083c8cc8184d)

## Model Comparison
We evaluated three state-of-the-art language models:

1. **Phi-2 (2.7B)**: Smallest model with fastest response times
2. **Mistral-7B (7B)**: Mid-sized model with balanced performance
3. **LLaMA 3 (8B)**: Largest model with highest accuracy but slower response times

### Performance Metrics
Our comprehensive evaluation measured:

- Factual accuracy
- Concept coverage
- Response speed
- Reasoning quality
- Context usage

![Performance Comparison](https://github.com/user-attachments/assets/9578f95f-5858-4042-a6fe-083c8cc8184d)

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/venureddy1718/EduRAG.git
cd EduRAG

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Processing Papers
```python
from code.api.arxiv import ArxivAPI

# Initialize API
arxiv_api = ArxivAPI(save_dir="data/papers")

# Search and download papers
papers = arxiv_api.search_and_download("Retrieval Augmented Generation", max_results=5)
```

### Running the RAG Pipeline
```python
from code.rag.pipeline import EduRAGPipeline

# Initialize pipeline
pipeline = EduRAGPipeline()

# Ask a question
answer = pipeline.answer_question("What are the key components of a RAG system?")
print(answer)
```

### Using the Streamlit App
```bash
# Start the Streamlit app
streamlit run code/app/streamlit_app.py
```

## Key Features

- Integration with academic paper repositories (ArXiv, Semantic Scholar)
- Intelligent chunking of research papers for optimal retrieval
- Vector-based similarity search using ChromaDB
- Support for multiple language models with different performance characteristics
- Interactive user interface for question answering
- Comprehensive evaluation framework for model comparison

## Results and Findings
Our evaluation revealed important trade-offs between model size, response quality, and performance:

- **Phi-2 (2.7B)**: Achieved the fastest response times (82.30s on average) but with lower factual accuracy (0.72) and reasoning scores (0.68). Best suited for applications where speed is critical.
- **Mistral-7B (7B)**: Demonstrated balanced performance with good reasoning capabilities (0.88) and moderate response times (108.84s on average). Recommended for general educational use cases.
- **LLaMA 3 (8B)**: Provided the highest factual accuracy (0.88) and concept coverage (0.95) but with the slowest response times (120.09s on average). Optimal for research scenarios where accuracy is paramount.

## Project Structure
```
EduRAG/
├── code/
│   ├── api/              # API integrations for paper sources
│   ├── retrieval/        # Document processing and chunking
│   ├── rag/              # Core RAG implementation
│   └── app/              # Streamlit application
├── evaluation/           # Evaluation results and metrics
│   ├── results/          # Visualization images
│   └── domain_specific_questions.json
├── docs/                 # Documentation
└── presentation/         # Presentation materials
```

## References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS.
2. Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain Question Answering." EMNLP.
3. Johnson, J., et al. (2017). "Billion-Scale Similarity Search with GPUs." IEEE Transactions on Big Data.
4. Borgeaud, S., et al. (2022). "Improving Language Models by Retrieving from Trillions of Tokens." ICML.
5. Asai, A., et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." arXiv.
6. Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP.
7. Ram, O., et al. (2023). "In-Context Retrieval-Augmented Language Models." TACL.

## Future Work

- Integrate multilingual support for papers in different languages
- Implement multimodal capabilities for figures and diagrams
- Develop domain-specific fine-tuning for specialized academic fields
- Improve response speed through model optimization techniques
- Add collaborative features for team research environments

## License
This project is available under the MIT License.

## Acknowledgments
This project was developed as part of advanced NLP research on improving academic content understanding.
