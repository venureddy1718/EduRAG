import os
import sys
import time
import json
import streamlit as st
from pathlib import Path

# Add project root to path for imports
import sys
project_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_base)

# Import EduRAG components
from api.arxiv import ArxivAPI
from retrieval.pdf_converter import PDFConverter
from retrieval.chunker import TextChunker
from rag.retriever import RAGRetriever
from rag.generator import RAGGenerator
from rag.pipeline import EduRAGPipeline

# Define paths
papers_dir = os.path.join(project_base, 'data/papers')
processed_dir = os.path.join(project_base, 'data/processed/text')
chunks_dir = os.path.join(project_base, 'data/processed/chunks')
embed_dir = os.path.join(project_base, 'data/embeddings')
evaluation_dir = os.path.join(project_base, 'data/evaluation')

# Ensure directories exist
for dir_path in [papers_dir, processed_dir, chunks_dir, embed_dir, evaluation_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="EduRAG - Academic Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("EduRAG: Academic Research Assistant")
st.markdown("""
This application uses Retrieval-Augmented Generation (RAG) to help understand academic research papers.
Upload papers, ask questions, and get informed responses based on the content of the papers.
""")

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'collection_initialized' not in st.session_state:
    st.session_state.collection_initialized = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'model_evaluation' not in st.session_state:
    st.session_state.model_evaluation = None

# Function to initialize components
def initialize_components():
    with st.spinner("Initializing components..."):
        # Initialize retriever
        st.session_state.retriever = RAGRetriever(
            persist_directory=embed_dir
        )
        
        # Check if we want to load a model or use a mock generator for demo
        use_real_model = st.session_state.get('use_real_model', False)
        
        if use_real_model:
            model_path = st.session_state.get('model_path', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
            
            # Initialize a real generator
            generator = RAGGenerator(
                model_name_or_path=model_path,
                use_quantization=False  # Set to True for larger models
            )
        else:
            # Use a mock generator for demo purposes
            class MockGenerator:
                def answer_from_context(self, query, context_chunks, system_prompt=None, 
                                       max_new_tokens=512, temperature=0.7):
                    sources = []
                    for chunk in context_chunks:
                        source = chunk.get('metadata', {}).get('paper_id', 'unknown')
                        if source not in sources:
                            sources.append(source)
                    
                    # Generate a mock answer based on retrieved content
                    source_text = [chunk.get('text', '') for chunk in context_chunks]
                    mock_answer = f"Based on the retrieved papers ({', '.join(sources)}), "
                    mock_answer += "Retrieval Augmented Generation (RAG) is a technique that enhances LLMs by "
                    mock_answer += "integrating external knowledge sources to provide more accurate, up-to-date, "
                    mock_answer += "and factually grounded responses."
                    
                    return mock_answer
            
            generator = MockGenerator()
        
        # Initialize other components
        arxiv_api = ArxivAPI(papers_dir)
        pdf_converter = PDFConverter(output_dir=processed_dir)
        text_chunker = TextChunker(output_dir=chunks_dir)
        
        # Initialize the pipeline
        st.session_state.pipeline = EduRAGPipeline(
            arxiv_api=arxiv_api,
            pdf_converter=pdf_converter,
            text_chunker=text_chunker,
            retriever=st.session_state.retriever,
            generator=generator
        )
        
        st.success("Components initialized successfully!")

# Sidebar for settings and controls
with st.sidebar:
    st.header("Settings")
    
    # Initialize components button
    if st.button("Initialize Components"):
        initialize_components()
    
    st.divider()
    
    # Model selection
    st.subheader("Model Selection")
    use_real_model = st.checkbox("Use Real LLM Model", value=False, 
                               help="Warning: This will load a real LLM model which requires more resources")
    
    if use_real_model:
        model_options = {
            "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Phi-2": "microsoft/phi-2",
            "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.1",
            "LLaMA-3": "meta-llama/Llama-3-8B-Instruct"
        }
        
        selected_model = st.selectbox("Select Model", list(model_options.keys()))
        st.session_state.model_path = model_options[selected_model]
        st.session_state.use_real_model = True
    else:
        st.session_state.use_real_model = False
    
    st.divider()
    
    # Search and process papers
    st.subheader("Add Papers")
    search_query = st.text_input("ArXiv Search Query")
    max_results = st.slider("Max Results", 1, 10, 3)
    
    if st.button("Search and Process Papers"):
        if st.session_state.pipeline is None:
            st.error("Please initialize components first!")
        else:
            with st.spinner("Searching and processing papers..."):
                processed_papers = st.session_state.pipeline.search_and_process_papers(
                    search_query, max_results=max_results
                )
                if processed_papers:
                    st.success(f"Successfully processed {len(processed_papers)} papers")
                    # Update retriever after processing new papers
                    st.session_state.pipeline.build_knowledge_base(chunks_dir)
                    st.session_state.collection_initialized = True
                else:
                    st.warning("No papers were processed")
    
    st.divider()
    
    # Build knowledge base
    st.subheader("Knowledge Base")
    
    if st.button("Build/Update Knowledge Base"):
        if st.session_state.pipeline is None:
            st.error("Please initialize components first!")
        else:
            with st.spinner("Building knowledge base..."):
                success = st.session_state.pipeline.build_knowledge_base(chunks_dir)
                if success:
                    st.success("Knowledge base built successfully")
                    st.session_state.collection_initialized = True
                else:
                    st.error("Failed to build knowledge base")
    
    # Collection status
    if st.session_state.collection_initialized:
        st.success("Knowledge base is ready")
    else:
        st.warning("Knowledge base not initialized")
    
    st.divider()
    
    # Model evaluation
    st.subheader("Model Evaluation")
    if st.button("Start Model Evaluation"):
        st.warning("Model evaluation would load multiple LLMs. This is a simulation.")
        
        # Create sample evaluation data
        st.session_state.model_evaluation = {
            "questions": [
                "What is retrieval augmented generation?",
                "How does RAG improve language models?",
                "What are the challenges in implementing RAG systems?"
            ],
            "models": [
                {"name": "Mistral-7B", "path": "mistralai/Mistral-7B-Instruct-v0.1"},
                {"name": "LLaMA-3", "path": "meta-llama/Llama-3-8B-Instruct"},
                {"name": "Phi-2", "path": "microsoft/phi-2"}
            ],
            "answers": {
                "Mistral-7B": [
                    {
                        "question": "What is retrieval augmented generation?",
                        "answer": "Retrieval Augmented Generation (RAG) is a technique that enhances Large Language Models by integrating external knowledge sources. It retrieves relevant documents to provide more accurate and up-to-date information.",
                        "sources": [{"paper_id": "paper1", "section": "abstract", "score": 0.95}]
                    },
                    # More sample results...
                ],
                "LLaMA-3": [
                    # Sample results...
                ],
                "Phi-2": [
                    # Sample results...
                ]
            }
        }
        
        st.success("Evaluation simulation created")

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Ask Questions", "Paper Management", "Evaluation"])

# Tab 1: Ask Questions
with tab1:
    # Chat interface
    st.header("Ask Questions about Academic Papers")
    
    # Info message if no knowledge base
    if not st.session_state.collection_initialized:
        st.info("Please initialize the knowledge base first to start asking questions.")
    
    # Display conversation history
    for i, (role, message) in enumerate(st.session_state.conversation_history):
        if role == "user":
            st.chat_message("user").write(message)
        else:
            with st.chat_message("assistant"):
                st.write(message.get('answer', ''))
                if 'sources' in message and message['sources']:
                    with st.expander("Sources"):
                        for source in message['sources']:
                            st.write(f"- {source.get('paper_id')} ({source.get('section', 'N/A')}): {source.get('score', 0.0):.4f}")
    
    # User input
    user_question = st.chat_input("Ask a question about academic papers")
    
    if user_question:
        # Add user message to chat
        st.chat_message("user").write(user_question)
        st.session_state.conversation_history.append(("user", user_question))
        
        # Check if pipeline is initialized
        if st.session_state.pipeline is None:
            with st.chat_message("assistant"):
                st.write("Please initialize components first!")
                st.session_state.conversation_history.append(("assistant", {"answer": "Please initialize components first!"}))
        # Check if knowledge base is initialized
        elif not st.session_state.collection_initialized:
            with st.chat_message("assistant"):
                st.write("Please build the knowledge base first!")
                st.session_state.conversation_history.append(("assistant", {"answer": "Please build the knowledge base first!"}))
        else:
            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer_result = st.session_state.pipeline.answer_question(user_question)
                    
                # Display answer
                st.write(answer_result.get('answer', ''))
                
                # Display sources if available
                if 'sources' in answer_result and answer_result['sources']:
                    with st.expander("Sources"):
                        for source in answer_result['sources']:
                            st.write(f"- {source.get('paper_id')} ({source.get('section', 'N/A')}): {source.get('score', 0.0):.4f}")
                
                # Add to conversation history
                st.session_state.conversation_history.append(("assistant", answer_result))

# Tab 2: Paper Management
with tab2:
    st.header("Paper Management")
    
    # List processed papers
    st.subheader("Processed Papers")
    
    # Get list of processed papers
    processed_papers = list(Path(processed_dir).glob("processed_*.json"))
    if processed_papers:
        paper_data = []
        for paper_file in processed_papers:
            try:
                with open(paper_file, 'r') as f:
                    data = json.load(f)
                    paper_data.append({
                        "id": data.get('filename', paper_file.stem.replace('processed_', '')),
                        "title": data.get('title', 'Unknown'),
                        "abstract": data.get('abstract', 'N/A')
                    })
            except:
                # Skip files that can't be read
                pass
        
        # Display paper data
        for i, paper in enumerate(paper_data):
            with st.expander(f"{i+1}. {paper['title'] or paper['id']}"):
                st.write(f"**ID:** {paper['id']}")
                st.write(f"**Abstract:** {paper['abstract']}")
    else:
        st.info("No processed papers found. Search for papers first.")
    
    # Manual paper upload
    st.subheader("Upload Paper PDF")
    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")
    
    if uploaded_file is not None:
        st.warning("PDF upload functionality would be implemented here.")
        st.info("In a full implementation, this would process the uploaded PDF through the RAG pipeline.")

# Tab 3: Model Evaluation
with tab3:
    st.header("Model Evaluation")
    
    if st.session_state.model_evaluation is not None:
        # Display evaluation results
        st.subheader("Evaluation Questions")
        for i, question in enumerate(st.session_state.model_evaluation['questions']):
            st.write(f"{i+1}. {question}")
        
        st.subheader("Model Comparison")
        
        # Create tabs for each model
        model_tabs = st.tabs([model['name'] for model in st.session_state.model_evaluation['models']])
        
        for i, model in enumerate(st.session_state.model_evaluation['models']):
            with model_tabs[i]:
                st.write(f"**Model:** {model['name']}")
                st.write(f"**Path:** {model['path']}")
                
                # Display answers from this model
                answers = st.session_state.model_evaluation['answers'].get(model['name'], [])
                for j, answer_data in enumerate(answers):
                    with st.expander(f"Q{j+1}: {answer_data['question']}"):
                        st.write("**Answer:**")
                        st.write(answer_data['answer'])
                        
                        st.write("**Sources:**")
                        for source in answer_data.get('sources', []):
                            st.write(f"- {source.get('paper_id')} ({source.get('section', 'N/A')}): {source.get('score', 0.0):.4f}")
        
        # Comparison charts would be added here in a full implementation
        st.subheader("Comparison Metrics")
        st.info("In a full implementation, this section would display charts comparing model performance metrics.")
    else:
        st.info("Start model evaluation from the sidebar to see results.")
    
    # Notes on metrics
    st.subheader("Evaluation Methodology")
    st.markdown("""
    In a complete implementation, models would be evaluated using:
    
    1. **ROUGE and BLEU Scores:** Comparison to reference summaries
    2. **Factual Accuracy:** Correctness of generated answers
    3. **Human Evaluation:** Readability, usability, and satisfaction
    4. **Response Speed:** Latency in retrieving and generating answers
    """)

# Instructions at the bottom
st.markdown("---")
st.markdown("""
**Instructions:**
1. Initialize components using the sidebar button
2. Search for papers or build the knowledge base
3. Ask questions in the chat interface
4. Explore processed papers and evaluation results in the other tabs
""")