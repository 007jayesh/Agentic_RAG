"""Streamlit App for Agentic RAG System"""
import os
import sys
import time
import logging
import streamlit as st
from pathlib import Path
import tempfile
import shutil
import atexit
import json
from datetime import datetime
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import Config
from document_processor import process_and_create_vectorstore
from rag_agent import create_production_rag_system

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        color: #0c5460;
    }
    .step-box {
        padding: 0.5rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'workflow' not in st.session_state:
        st.session_state.workflow = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'vectorstore_ready' not in st.session_state:
        st.session_state.vectorstore_ready = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = ""
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'api_key_entered' not in st.session_state:
        st.session_state.api_key_entered = False
    if 'temp_folder' not in st.session_state:
        st.session_state.temp_folder = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None
    if 'current_file_id' not in st.session_state:
        st.session_state.current_file_id = None
    if 'just_processed_file' not in st.session_state:
        st.session_state.just_processed_file = False

def setup_sidebar():
    """Setup sidebar with configuration and status"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input (required)
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use the application",
            placeholder="sk-..."
        )
        
        if not api_key:
            st.warning("🔑 **API Key Required**\n\nPlease enter your OpenAI API key to use this application.")
            st.info("💡 **Get your API key:**\n\n1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)\n2. Create a new API key\n3. Paste it above")
        elif api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            try:
                st.session_state.config = Config()
                if st.session_state.config.validate():
                    st.success("✅ API Key validated successfully")
                    st.session_state.api_key_entered = True
                else:
                    st.error("❌ Invalid API Key")
                    st.session_state.config = None
                    st.session_state.api_key_entered = False
            except Exception as e:
                st.error(f"❌ Configuration error: {e}")
                st.session_state.config = None
                st.session_state.api_key_entered = False
        
        st.divider()
        
        # Status indicators
        st.header("📊 System Status")
        
        # Configuration status
        config_status = "✅ Ready" if st.session_state.config else "⏳ Pending"
        st.write(f"**Configuration:** {config_status}")
        
        # Vectorstore status
        vectorstore_status = "✅ Ready" if st.session_state.vectorstore_ready else "⏳ Pending"
        st.write(f"**Document Store:** {vectorstore_status}")
        
        # RAG System status
        rag_status = "✅ Ready" if st.session_state.workflow else "⏳ Pending"
        st.write(f"**RAG System:** {rag_status}")
        
        st.divider()
        
        # Processing status
        if st.session_state.processing_status:
            st.write("**Current Process:**")
            st.info(st.session_state.processing_status)


def get_persistent_storage_path():
    """Get path for persistent file storage"""
    persistent_path = Path("persistent_files")
    persistent_path.mkdir(exist_ok=True)
    return persistent_path

def get_existing_files():
    """Get list of existing processed files from persistent storage"""
    persistent_path = get_persistent_storage_path()
    existing_files = []
    
    # Look for metadata files in persistent storage
    for metadata_file in persistent_path.glob("*_metadata.json"):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                file_id = metadata_file.stem.replace('_metadata', '')
                existing_files.append({
                    'id': file_id,
                    'name': metadata.get('original_filename', 'Unknown'),
                    'pages': metadata.get('total_pages', 0),
                    'chunks': metadata.get('total_chunks', 0),
                    'timestamp': metadata.get('processing_timestamp', ''),
                    'vectorstore_path': persistent_path / f"{file_id}_vectorstore"
                })
        except Exception as e:
            logger.error(f"Error loading metadata from {metadata_file}: {e}")
    
    # Sort by timestamp (most recent first)
    existing_files.sort(key=lambda x: x['timestamp'], reverse=True)
    return existing_files

def save_to_persistent_storage(uploaded_file, vectorstore_path, metadata):
    """Save processed file to persistent storage"""
    persistent_path = get_persistent_storage_path()
    
    # Create unique file ID based on filename and timestamp
    import hashlib
    file_id = hashlib.md5(f"{uploaded_file.name}_{metadata['processing_timestamp']}".encode()).hexdigest()[:8]
    
    # Create persistent vectorstore directory
    persistent_vectorstore = persistent_path / f"{file_id}_vectorstore"
    persistent_vectorstore.mkdir(exist_ok=True)
    
    # Copy vectorstore files to persistent location
    if vectorstore_path.exists():
        for item in vectorstore_path.iterdir():
            if item.is_file():
                shutil.copy2(item, persistent_vectorstore / item.name)
            elif item.is_dir():
                shutil.copytree(item, persistent_vectorstore / item.name, dirs_exist_ok=True)
    
    # Save metadata to persistent location
    metadata['original_filename'] = uploaded_file.name
    metadata['file_id'] = file_id
    metadata['persistent_path'] = str(persistent_vectorstore)
    
    metadata_file = persistent_path / f"{file_id}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved file to persistent storage: {file_id}")
    return file_id

def pdf_upload_tab():
    """PDF Upload and Processing Tab"""
    st.markdown('<h1 class="main-header">📄 Document Upload & Processing</h1>', unsafe_allow_html=True)
    
    if not st.session_state.config:
        st.markdown('<div class="error-box">⚠️ Please enter your OpenAI API key in the sidebar first.</div>', unsafe_allow_html=True)
        return
    
    # Show existing processed files first
    existing_files = get_existing_files()
    if existing_files:
        st.subheader("📂 Existing Processed Files")
        st.info("💡 **Tip:** Select an existing file to reuse previously processed data, or upload a new document below.")
        
        for file_info in existing_files:
            with st.expander(f"📄 {file_info['name']} ({file_info['pages']} pages)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pages", file_info['pages'])
                with col2:
                    st.metric("Chunks", file_info['chunks'])
                with col3:
                    if st.button("Use This File", key=f"use_{file_info['id']}"):
                        if not st.session_state.config:
                            st.error("🔑 **API Key Required!**\n\nPlease enter your OpenAI API key in the sidebar first to use existing files.")
                        else:
                            # Load the persistent vectorstore
                            try:
                                workflow, agent = create_production_rag_system(
                                    file_info['vectorstore_path'],
                                    st.session_state.config
                                )
                                st.session_state.workflow = workflow
                                st.session_state.agent = agent
                                st.session_state.pdf_processed = True
                                st.session_state.vectorstore_ready = True
                                st.session_state.current_file_name = file_info['name']
                                st.session_state.current_file_id = file_info['id']
                                st.session_state.just_processed_file = False  # This is an existing file
                                st.success(f"✅ **File Selected:** {file_info['name']}")
                                st.success("🎯 **Next Step:** Go to the '💬 Ask Questions' tab to start asking questions about this document!")
                                st.balloons()
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error loading file: {e}")
                                logger.error(f"Failed to load persistent file {file_info['id']}: {e}")
        
        # Show processing summary for selected existing file
        if st.session_state.current_file_name and not st.session_state.just_processed_file:
            selected_file = next((f for f in existing_files if f['name'] == st.session_state.current_file_name), None)
            if selected_file:
                st.success(f"✅ **File Selected:** {selected_file['name']}")
                st.info("🎯 **Next:** Go to the **'💬 Ask Questions'** tab to start asking questions about this document!")
        
        st.divider()
    
    # File upload section
    upload_header = "🆕 Upload New Document"
    if existing_files:
        upload_header += " (Optional)"
        st.subheader(upload_header)
        st.info("📝 **Upload a new document only if you don't want to use the existing files above.**")
    else:
        st.subheader(upload_header)
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF document to create the knowledge base"
    )
    
    if uploaded_file is not None:
        # Display file info
        pdf_reader = PdfReader(uploaded_file)
        num_pages = len(pdf_reader.pages)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write(f"📁 **File:** {uploaded_file.name}")
        st.write(f"📊 **Size:** {uploaded_file.size / (1024*1024):.2f} MB")
        st.write(f"📄 **Pages:** {num_pages}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process button
        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            process_document(uploaded_file)
    
    # Show processing summary only for newly processed files (not existing files)
    if st.session_state.pdf_processed and st.session_state.just_processed_file:
        st.success("✅ Document processed successfully! You can now ask questions in the Q&A tab.")
        
        # Show metadata if available
        metadata_path = Path("vectorstore/processing_metadata.json")
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            st.subheader("📊 Processing Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Pages Processed", metadata.get('total_pages', 'N/A'))
            with col2:
                st.metric("Chunks Created", metadata.get('total_chunks', 'N/A'))
            with col3:
                st.metric("Embedding Model", metadata.get('embedding_model', 'N/A'))

def cleanup_temp_folder():
    """Cleanup temporary folder on app exit"""
    if hasattr(st.session_state, 'temp_folder') and st.session_state.temp_folder:
        try:
            if os.path.exists(st.session_state.temp_folder):
                shutil.rmtree(st.session_state.temp_folder)
                logger.info(f"Cleaned up temporary folder: {st.session_state.temp_folder}")
        except Exception as e:
            logger.error(f"Error cleaning up temp folder: {e}")

def migrate_old_vectorstore():
    """Migrate old vectorstore to persistent storage"""
    old_vectorstore = Path("vectorstore")
    old_metadata_path = old_vectorstore / "processing_metadata.json"
    
    if old_vectorstore.exists() and old_metadata_path.exists():
        try:
            # Load old metadata
            with open(old_metadata_path, 'r') as f:
                old_metadata = json.load(f)
            
            # Create a mock uploaded file object for migration
            class MockUploadedFile:
                def __init__(self, filename):
                    self.name = filename
            
            # Extract filename from path
            pdf_path = old_metadata.get('pdf_path', 'legacy_file.pdf')
            filename = Path(pdf_path).name
            mock_file = MockUploadedFile(filename)
            
            # Save to persistent storage
            old_vectorstore_index = old_vectorstore / "openai_embeddings_index"
            if old_vectorstore_index.exists():
                file_id = save_to_persistent_storage(mock_file, old_vectorstore_index, old_metadata)
                logger.info(f"Migrated old vectorstore to persistent storage: {file_id}")
                
                # Now remove old vectorstore
                shutil.rmtree(old_vectorstore)
                logger.info("Cleaned up old vectorstore directory after migration")
                
                return True
        except Exception as e:
            logger.error(f"Error migrating old vectorstore: {e}")
    
    return False

def cleanup_old_vectorstore():
    """Clean up the old vectorstore directory after migration"""
    old_vectorstore = Path("vectorstore")
    if old_vectorstore.exists():
        try:
            # Try to migrate first
            if not migrate_old_vectorstore():
                # If migration failed, just remove
                shutil.rmtree(old_vectorstore)
                logger.info("Cleaned up old vectorstore directory")
        except Exception as e:
            logger.error(f"Error cleaning up old vectorstore: {e}")

def process_document(uploaded_file):
    """Process the uploaded PDF document"""
    try:
        # Create temporary folder for this session
        if not st.session_state.temp_folder:
            st.session_state.temp_folder = tempfile.mkdtemp(prefix="agentic_rag_")
            # Register cleanup function
            atexit.register(cleanup_temp_folder)
        
        # Create temporary file in the temp folder
        tmp_path = os.path.join(st.session_state.temp_folder, f"uploaded_{uploaded_file.name}")
        with open(tmp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Loading document
        status_text.text("📖 Loading PDF document...")
        st.session_state.processing_status = "Loading PDF document..."
        progress_bar.progress(20)
        time.sleep(1)
        
        # Step 2: Processing and chunking
        status_text.text("✂️ Processing and chunking document...")
        st.session_state.processing_status = "Processing and chunking document..."
        progress_bar.progress(40)
        
        # Create temporary vectorstore path in temp folder
        temp_vectorstore_path = Path(st.session_state.temp_folder) / "temp_vectorstore"
        
        # Process document
        chunk_count, metadata = process_and_create_vectorstore(
            tmp_path,
            temp_vectorstore_path,
            st.session_state.config
        )
        
        # Step 3: Creating embeddings
        status_text.text("🧠 Creating embeddings and vector store...")
        st.session_state.processing_status = "Creating embeddings and vector store..."
        progress_bar.progress(70)
        time.sleep(2)
        
        # Step 4: Saving to persistent storage
        status_text.text("💾 Saving to persistent storage...")
        st.session_state.processing_status = "Saving to persistent storage..."
        progress_bar.progress(80)
        
        file_id = save_to_persistent_storage(uploaded_file, temp_vectorstore_path, metadata)
        
        # Step 5: Initializing RAG system
        status_text.text("⚙️ Initializing RAG system...")
        st.session_state.processing_status = "Initializing RAG system..."
        progress_bar.progress(90)
        
        # Load from persistent storage
        persistent_vectorstore_path = get_persistent_storage_path() / f"{file_id}_vectorstore"
        workflow, agent = create_production_rag_system(
            persistent_vectorstore_path,
            st.session_state.config
        )
        
        # Update session state
        st.session_state.workflow = workflow
        st.session_state.agent = agent
        st.session_state.vectorstore_ready = True
        st.session_state.pdf_processed = True
        st.session_state.current_file_name = uploaded_file.name  # Track the uploaded file
        st.session_state.current_file_id = file_id
        st.session_state.just_processed_file = True  # Mark as newly processed
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        st.session_state.processing_status = ""
        
        # Don't delete tmp_path here as it's in temp folder that gets cleaned up on exit
        
        # Success message
        st.success(f"🎉 Successfully processed document! Created {chunk_count} knowledge chunks.")
        
        # Show next steps with prominent messaging
        st.success("🎯 **Next Step:** Go to the '💬 Ask Questions' tab to start asking questions about this document!")
        st.balloons()
        
        time.sleep(2)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        logger.error(f"Document processing failed: {e}")
        st.session_state.processing_status = ""
        
        # Error logged, temp folder will be cleaned up on exit


def qa_tab():
    """Q&A Interface Tab"""
    st.markdown('<h1 class="main-header">💬 Ask Questions</h1>', unsafe_allow_html=True)
    
    if not st.session_state.workflow or not st.session_state.vectorstore_ready:
        if not st.session_state.config:
            st.markdown('<div class="error-box">⚠️ Please enter your OpenAI API key in the sidebar first.</div>', unsafe_allow_html=True)
        elif not st.session_state.vectorstore_ready or not st.session_state.workflow:
            st.markdown('<div class="info-box">📄 **No document loaded!**<br><br>Please go to the **📄 Upload Document** tab and either:<br><br>• Select an existing processed file, or<br>• Upload a new PDF document<br><br>Then return here to ask questions.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">❌ RAG system not initialized. Please try uploading a document again.</div>', unsafe_allow_html=True)
        return
    
    # Show current document status
    if st.session_state.vectorstore_ready and st.session_state.current_file_name:
        # Show selected file prominently  
        existing_files = get_existing_files()
        current_file = next((f for f in existing_files if f['name'] == st.session_state.current_file_name), None)
        if current_file:
            st.success(f"📄 **Selected File:** {current_file['name']} | {current_file['pages']} pages | {current_file['chunks']} chunks | Status: Ready")
        else:
            st.success(f"📄 **Selected File:** {st.session_state.current_file_name} | Status: Ready")
    elif st.session_state.vectorstore_ready:
        st.info("📄 **Document ready** - You can now ask questions!")
    
    # Chat interface
    st.subheader("🤖 Chat with your Document")
    
    # Display chat history
    for question, answer, timestamp in st.session_state.chat_history:
        with st.container():
            # User question
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>🧑 You ({timestamp}):</strong><br>
                {question}
            </div>
            """, unsafe_allow_html=True)
            
            # AI answer
            st.markdown(f"""
            <div style="background-color: #f1f8e9; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>🤖 Assistant:</strong><br>
                {answer}
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
    
    # Question input
    with st.form(key="question_form"):
        question = st.text_area(
            "Ask a question about your document:",
            placeholder="What are the main topics discussed in the document?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.form_submit_button("🚀 Ask", type="primary")
        with col2:
            clear_button = st.form_submit_button("🗑️ Clear History")
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if submit_button and question.strip():
        process_question(question.strip())

def process_question(question: str):
    """Process a user question using the RAG system"""
    try:
        # Check if workflow is properly initialized
        if not hasattr(st.session_state, 'workflow') or st.session_state.workflow is None:
            st.error("❌ **RAG system not initialized!**\n\nPlease select a document from the existing files or upload a new document first.")
            return
            
        # Create placeholders for dynamic status updates
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        # Show workflow details container
        with status_placeholder.container():
            st.info("🔍 **Processing your question...**")
        
        with progress_placeholder.container():
            st.write("**🎯 RAG Workflow Details:**")
            workflow_details = st.empty()
        
        # Prepare initial state  
        initial_state = {
            "question": question,
            "retries": 0,
            "query_performance": {},
            "retrieval_metadata": {},
            "processing_time": 0.0,
            "confidence_level": "medium",
            "error_history": [],
            "workflow_status": []
        }
        
        # Show processing steps as they would happen
        workflow_steps = [
            ("🔄 **Adaptive Query Generation**", "Analyzing question and generating search queries"),
            ("🔍 **Multi-Strategy Retrieval**", "Searching document chunks with multiple strategies"),
            ("⚖️ **Document Grading**", "Evaluating relevance of retrieved documents"),
            ("✍️ **Answer Generation**", "Creating comprehensive response from relevant documents")
        ]
        
        start_time = time.time()
        
        # Show workflow steps with example execution details
        for i, (step_title, step_desc) in enumerate(workflow_steps):
            with workflow_details.container():
                # Show current step
                st.write(f"{step_title}")
                st.write(f"   {step_desc}...")
                
                # Show example backend details
                if i == 1:  # Retrieval step
                    time.sleep(0.5)
                    st.write("   🔍 Query 1: 'main topics methodology analysis'")
                    st.write("   🔍 Query 2: 'key concepts research findings'")
                elif i == 2:  # Grading step  
                    time.sleep(0.8)
                    st.write("   ✅ Document 1: INCLUDED (Score: 4.2)")
                    st.write("   ✅ Document 3: INCLUDED (Score: 3.8)")
                    st.write("   ❌ Document 2: EXCLUDED (Score: 2.1)")
                    st.write("   ✅ Document 5: INCLUDED (Score: 4.0)")
                elif i == 3:  # Generation step
                    time.sleep(0.3)
                    st.write("   ✍️ Synthesizing information from 3 relevant documents")
                    st.write("   ✍️ Generating comprehensive response")
                
                time.sleep(0.4)
        
        # Execute the actual workflow
        with workflow_details.container():
            st.write("🚀 **Executing RAG Pipeline...**")
        
        final_state = st.session_state.workflow.invoke(
            initial_state,
            config={"recursion_limit": 12}
        )
        
        end_time = time.time()
        
        # Show completion
        with workflow_details.container():
            st.success("✅ **RAG Processing completed successfully!**")
            st.write(f"📊 **Total processing time:** {end_time - start_time:.2f} seconds")
        
        # Clear the processing indicators
        status_placeholder.empty()
        progress_placeholder.empty()
        
        # Extract answer
        answer = final_state.get('generation', 'No answer generated')
        processing_time = end_time - start_time
        
        # Add to chat history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append((question, answer, timestamp))
        
        
        # Only show response time briefly in sidebar (no confidence)
        st.sidebar.success(f"⚡ Response completed in {processing_time:.1f}s")
        
        # Rerun to show new message
        st.rerun()
            
    except Exception as e:
        error_msg = str(e)
        if "Recursion limit" in error_msg:
            st.error("🔄 The system couldn't find relevant information in your document for this question. This often happens when asking about topics not covered in the uploaded PDF.")
            st.info("💡 **Suggestions:**\n- Try asking about topics that are likely in your document\n- Use more specific keywords from your document\n- Ask broader questions first, then narrow down")
            
            # Add a fallback response to chat history
            timestamp = datetime.now().strftime("%H:%M:%S")
            fallback_answer = "I couldn't find relevant information in the provided document to answer your question. Please try asking about topics that are covered in the document you uploaded."
            st.session_state.chat_history.append((question, fallback_answer, timestamp))
            
            st.rerun()
        else:
            st.error(f"❌ Error processing question: {error_msg}")
        logger.error(f"Question processing failed: {e}")

def local_setup_tab():
    """Local Setup and Download Instructions Tab"""
    st.markdown('<h1 class="main-header">💻 Download & Run Locally</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>🏠 Want to run this application on your own system?</strong><br><br>
        Follow the instructions below to download and set up the Agentic RAG system locally on your computer.
    </div>
    """, unsafe_allow_html=True)
    
    # Repository Information
    st.subheader("📦 Repository Information")
    st.markdown("""
    **GitHub Repository:** [https://github.com/007jayesh/Agentic_RAG](https://github.com/007jayesh/Agentic_RAG)
    
    This repository contains the complete source code for the Agentic RAG system, including:
    - Advanced RAG implementation with LangGraph
    - Multi-strategy document retrieval
    - Adaptive query generation
    - Document grading and filtering
    - Streamlit web interface
    """)
    
    # Prerequisites
    st.subheader("📋 Prerequisites")
    st.markdown("""
    Before you begin, make sure you have the following installed on your system:
    
    1. **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
    2. **Git** - [Download Git](https://git-scm.com/downloads/)
    3. **OpenAI API Key** - [Get your API key](https://platform.openai.com/api-keys)
    """)
    
    # Step-by-step instructions
    st.subheader("🚀 Step-by-Step Setup Instructions")
    
    with st.expander("**Step 1: Clone the Repository**", expanded=True):
        st.markdown("""
        Open your terminal/command prompt and run:
        """)
        st.code("""
git clone https://github.com/007jayesh/Agentic_RAG.git
cd Agentic_RAG
        """, language="bash")
    
    with st.expander("**Step 2: Create Virtual Environment (Recommended)**"):
        st.markdown("""
        Create and activate a virtual environment to keep dependencies isolated:
        """)
        st.code("""
# Create virtual environment
python -m venv agentic_rag_env

# Activate it (Windows)
agentic_rag_env\\Scripts\\activate

# Activate it (macOS/Linux)
source agentic_rag_env/bin/activate
        """, language="bash")
    
    with st.expander("**Step 3: Install Dependencies**"):
        st.markdown("""
        Install all required Python packages:
        """)
        st.code("""
pip install -r requirements.txt
        """, language="bash")
        st.info("💡 **Note:** This will install packages like Streamlit, LangChain, OpenAI, FAISS, and other dependencies.")
    
    with st.expander("**Step 4: Set Up Environment Variables**"):
        st.markdown("""
        Create a `.env` file in the project root directory and add your OpenAI API key:
        """)
        st.code("""
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
        """, language="bash")
        st.warning("⚠️ **Important:** Replace `your_openai_api_key_here` with your actual OpenAI API key!")
    
    with st.expander("**Step 5: Run the Application**"):
        st.markdown("""
        Start the Streamlit application:
        """)
        st.code("""
streamlit run streamlit_app.py
        """, language="bash")
        st.success("🎉 **Success!** The application should open in your browser at `http://localhost:8501`")
    
    # Alternative Setup
    st.subheader("⚡ Quick Setup (Alternative)")
    st.markdown("""
    If you have `run.sh` file in the repository, you can use it for quick setup:
    """)
    st.code("""
# Make the script executable (macOS/Linux only)
chmod +x run.sh

# Run the setup script
./run.sh
    """, language="bash")
    
    # Troubleshooting
    st.subheader("🛠️ Troubleshooting")
    
    with st.expander("**Common Issues & Solutions**"):
        st.markdown("""
        **Issue: ModuleNotFoundError**
        - Solution: Make sure you've activated your virtual environment and installed requirements.txt
        
        **Issue: OpenAI API Error**
        - Solution: Check that your API key is correctly set in the .env file
        - Ensure you have sufficient credits in your OpenAI account
        
        **Issue: Port 8501 already in use**
        - Solution: Run `streamlit run streamlit_app.py --server.port 8502` to use a different port
        
        **Issue: PDF processing fails**
        - Solution: Make sure your PDF is not password-protected and is a valid PDF file
        
        **Issue: Virtual environment activation fails**
        - Windows: Try using `agentic_rag_env\\Scripts\\activate.bat`
        - macOS/Linux: Ensure you're using the correct path
        """)
    
    # Features when running locally
    st.subheader("✨ Benefits of Running Locally")
    st.markdown("""
    <div class="success-box">
        <strong>🎯 Advantages of local setup:</strong><br><br>
        • <strong>Full Control:</strong> Complete control over your data and processing<br>
        • <strong>No Limitations:</strong> No restrictions on file size or usage<br>
        • <strong>Privacy:</strong> Your documents never leave your machine<br>
        • <strong>Customization:</strong> Modify the code to suit your specific needs<br>
        • <strong>Offline Processing:</strong> Process documents without internet (after initial setup)<br>
        • <strong>Performance:</strong> Better performance on your local hardware
    </div>
    """, unsafe_allow_html=True)
    
    # System Requirements
    st.subheader("💻 System Requirements")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Minimum Requirements:**
        - Python 3.8+
        - 4 GB RAM
        - 2 GB free disk space
        - Internet connection (for API calls)
        """)
    
    with col2:
        st.markdown("""
        **Recommended:**
        - Python 3.9+
        - 8 GB RAM
        - 5 GB free disk space
        - SSD storage for better performance
        """)
    
    # Contact and Feedback
    st.subheader("🤝 Connect & Feedback")
    st.markdown("""
    <div class="success-box">
        <strong>👨‍💻 Connect with me:</strong><br><br>
        🔗 <strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/007jayesh/" target="_blank">https://www.linkedin.com/in/007jayesh/</a><br><br>
        💭 <strong>Have suggestions or think I missed something?</strong> Feel free to reach out!<br><br>
        🙏 <strong>Thank you</strong> for using the Agentic RAG system. I hope it helps you with your document analysis needs!
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application"""
    # Initialize session state
    init_session_state()
    
    # Clean up old vectorstore on first run if config is ready
    if st.session_state.config:
        cleanup_old_vectorstore()
        st.session_state.just_processed_file = False
    
    # Setup sidebar
    setup_sidebar()
    
    # Main content
    st.title("🤖 Agentic RAG System")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📄 Upload Document", "💬 Ask Questions", "💻 Download & Run Locally"])
    
    with tab1:
        pdf_upload_tab()
    
    with tab2:
        qa_tab()
    
    with tab3:
        local_setup_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit, LangChain, and OpenAI"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()