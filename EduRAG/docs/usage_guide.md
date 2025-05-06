# EduRAG User Manual

## Introduction

EduRAG is a Retrieval Augmented Generation (RAG) system designed to assist students and researchers in comprehending complex academic papers. This user manual provides instructions on how to set up, configure, and use the EduRAG system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Data Sources](#data-sources)
4. [Using the Command Line Interface](#using-the-command-line-interface)
5. [Using the Streamlit Web Interface](#using-the-streamlit-web-interface)
6. [Choosing a Language Model](#choosing-a-language-model)
7. [Features and Capabilities](#features-and-capabilities)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

## System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+ recommended)
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 8GB, 16GB recommended
- **Storage**: At least 10GB of free space
- **GPU**: Optional but recommended for faster processing (CUDA-compatible)
- **Internet Connection**: Required for API access and model downloads

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/EduRAG.git
   cd EduRAG
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv env
   
   # On Windows
   env\Scripts\activate
   
   # On macOS/Linux
   source env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights** (if not using API endpoints):
   ```bash
   # Run the model setup script
   python -m code.setup_models
   ```

## Data Sources

EduRAG can fetch papers from multiple sources:

### ArXiv
Papers are fetched using the official ArXiv API. No authentication is required.

### Semantic Scholar
Academic papers are retrieved using the Semantic Scholar API. No API key is required for basic usage, but for higher rate limits:

1. Visit [Semantic Scholar API](https://www.semanticscholar.org/product/api)
2. Register for an API key
3. Add the key to your environment variables:
   ```bash
   # On Windows
   set SEMANTIC_SCHOLAR_API_KEY=your_api_key
   
   # On macOS/Linux
   export SEMANTIC_SCHOLAR_API_KEY=your_api_key
   ```

### Local PDF Files
You can also process locally stored PDF files:

1. Place PDF files in the `data/papers/` directory
2. Use the document loader function to process these files

## Using the Command Line Interface

EduRAG provides a simple command line interface for basic operations:

### Search and Download Papers

```python
from code.api.arxiv import ArxivAPI

# Initialize API
arxiv_api = ArxivAPI(save_dir="data/papers")

# Search and download papers
papers = arxiv_api.search_and_download("Retrieval Augmented Generation", max_results=5)
```

### Process a Specific Paper

```python
from code.retrieval.document_loader import PDFLoader
from code.rag.pipeline import EduRAGPipeline

# Load a PDF
loader = PDFLoader()
document = loader.load("data/papers/my_paper.pdf")

# Initialize pipeline with default model (Mistral-7B)
pipeline = EduRAGPipeline()

# Get a summary
summary = pipeline.generate_summary(document)
print(summary)
```

### Ask Questions About a Paper

```python
# Initialize pipeline
pipeline = EduRAGPipeline()

# Ask a question about a specific paper
answer = pipeline.answer_question(
    question="What methodology did the authors use?",
    paper_id="2307.15293"  # ArXiv ID or local file path
)
print(answer)
```

## Using the Streamlit Web Interface

The Streamlit app provides a user-friendly interface for interacting with EduRAG:

1. **Start the app**:
   ```bash
   streamlit run code/app/streamlit_app.py
   ```

2. **Access the interface** by opening a browser and navigating to `http://localhost:8501`

3. **Interface Features**:
   - **Search Tab**: Search for papers by keyword, author, or title
   - **Upload Tab**: Upload your own PDF files
   - **Summary Tab**: Generate concise summaries of selected papers
   - **Q&A Tab**: Ask specific questions about paper content
   - **Model Selection**: Choose between Phi-2, Mistral-7B, and LLaMA 3 models
   - **Settings**: Configure model parameters (temperature, max tokens, etc.)

## Choosing a Language Model

EduRAG supports three language models, each with different characteristics:

### Phi-2 (2.7B parameters)
- **Pros**: Fastest response times, lowest resource requirements
- **Cons**: Lower factual accuracy and reasoning quality
- **Best for**: Quick information lookup, resource-constrained environments
- **Average response time**: 82.30s

### Mistral-7B (7B parameters)
- **Pros**: Balanced performance, good reasoning capabilities
- **Cons**: Moderate resource requirements
- **Best for**: General educational use cases, everyday research tasks
- **Average response time**: 108.84s

### LLaMA 3 (8B parameters)
- **Pros**: Highest factual accuracy and concept coverage
- **Cons**: Slowest response times, highest resource requirements
- **Best for**: Detailed research work where accuracy is paramount
- **Average response time**: 120.09s

To select a model in the command line:

```python
from code.rag.pipeline import EduRAGPipeline

# Initialize with specific model
pipeline = EduRAGPipeline(model_name="llama3")  # Options: "phi2", "mistral7b", "llama3"
```

## Features and Capabilities

EduRAG provides several key features:

### Paper Summaries
Generate concise summaries of entire papers or specific sections.

### Concept Explanation
Ask about specific concepts, methodologies, or findings mentioned in a paper.

### Literature Comparison
Compare multiple papers on similar topics to identify differences and similarities.

### Citation Generation
Generate properly formatted citations for papers in various academic styles.

### Figure and Table Extraction
Extract and explain figures, tables, and illustrations from papers.

## Troubleshooting

### Common Issues

#### "Out of Memory" Errors
- Try using a smaller model (Phi-2)
- Reduce batch size in configuration
- Process fewer papers simultaneously

#### Slow Response Times
- Enable GPU acceleration if available
- Reduce chunk size for document processing
- Use the Phi-2 model for faster responses

#### Model Download Failures
- Check internet connection
- Ensure sufficient disk space
- Try downloading models manually from HuggingFace

#### API Rate Limiting
- Implement delays between requests
- Register for API keys when possible
- Use local PDF files instead of API downloads

### Logging
Check the logs in the `logs/` directory for detailed error information.

## FAQ

**Q: Can EduRAG work offline?**
A: Yes, once models are downloaded and papers are stored locally.

**Q: How many papers can be processed simultaneously?**
A: Depends on your system resources. Typically 5-10 papers for 16GB RAM.

**Q: Are the models running locally or in the cloud?**
A: By default, all models run locally on your machine for privacy.

**Q: Can I fine-tune the models on my own data?**
A: Yes, though this requires additional setup. See the advanced configuration guide.

**Q: What languages are supported?**
A: Currently English only, with plans for multilingual support.

**Q: How accurate are the answers?**
A: Accuracy varies by model, with LLaMA 3 achieving 88% factual accuracy in our tests.

---

For additional help or to report issues, please open an issue on our GitHub repository.
