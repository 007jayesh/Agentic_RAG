# 🤖 Agentic RAG System with Streamlit

A production-ready Retrieval-Augmented Generation (RAG) system built with Streamlit, LangChain, and OpenAI. Upload PDF documents and ask intelligent questions using advanced agentic workflows.

## 🚀 Features

- **📄 PDF Processing**: Upload and process PDF documents with advanced chunking
- **🧠 Intelligent Retrieval**: Multi-strategy document retrieval with semantic search
- **🤖 Agentic Workflows**: Smart query generation and document grading
- **💬 Chat Interface**: Interactive Q&A with conversation history
- **⚡ Real-time Progress**: Live updates during document processing
- **🎯 Confidence Scoring**: Quality assessment for generated responses

## 🛠️ Installation

1. **Clone and navigate to the project directory**
   ```bash
   cd /path/to/your/project
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv agentic_rag_env
   source agentic_rag_env/bin/activate  # On Windows: agentic_rag_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.template .env
   # Edit .env and add your OpenAI API key
   ```

## 🔑 Configuration

### OpenAI API Key
You need an OpenAI API key to use this system. You can either:
1. Set it in a `.env` file (recommended for local development)
2. Enter it directly in the Streamlit sidebar when running the app

### Environment Variables
```bash
OPENAI_API_KEY=your_api_key_here
```

## 🚀 Usage

1. **Start the application**
   ```bash
   source agentic_rag_env/bin/activate
   streamlit run streamlit_app.py
   ```

2. **Configure your API key**
   - Enter your OpenAI API key in the sidebar
   - The system will validate the key automatically

3. **Upload a PDF document**
   - Go to the "Upload Document" tab
   - Upload your PDF file
   - Watch the real-time progress as the document is processed

4. **Ask questions**
   - Switch to the "Ask Questions" tab
   - Enter your questions about the document
   - View responses with confidence scores

## 📋 System Requirements

- Python 3.9+
- OpenAI API key
- At least 4GB RAM (for larger documents)
- Internet connection for API calls

## 🏗️ Architecture

### Core Components

1. **config.py** - Configuration management and validation
2. **document_processor.py** - PDF processing and chunking logic
3. **rag_agent.py** - Agentic RAG workflows and LangGraph implementation
4. **streamlit_app.py** - Web interface and user interactions

### Processing Flow

```
PDF Upload → Text Extraction → Cleaning → Chunking → Embeddings → Vector Store
                                                                        ↓
Question → Query Generation → Multi-Strategy Retrieval → Document Grading → Answer Generation
```

## 🔧 Advanced Features

### Multi-Strategy Retrieval
- **Similarity Search**: Direct semantic matching
- **MMR (Maximal Marginal Relevance)**: Diverse results
- **Score Thresholding**: Quality filtering

### Agentic Workflows
- **Adaptive Query Generation**: Different strategies based on retry count
- **Intelligent Document Grading**: Quality assessment of retrieved documents
- **Enhanced Fallback**: Helpful suggestions when information isn't found

### Progress Tracking
Real-time updates for:
- Document loading
- Text processing and cleaning
- Chunk creation
- Embedding generation
- Vector store creation

## 📊 Performance Tips

1. **Document Size**: Larger documents take more time to process
2. **API Limits**: Be mindful of OpenAI rate limits
3. **Chunk Size**: Adjust in config.py based on your documents
4. **Question Quality**: Specific questions get better results

## 🛡️ Security

- API keys are handled securely
- No data is stored permanently on disk
- All processing happens locally except for OpenAI API calls

## 🐛 Troubleshooting

### Common Issues

1. **"API key not found"**
   - Ensure your OpenAI API key is properly set
   - Check the .env file or sidebar input

2. **"Failed to load vectorstore"**
   - Make sure you've processed a document first
   - Check that the vectorstore directory exists

3. **Slow processing**
   - Large PDFs take more time
   - Check your internet connection
   - Consider upgrading your OpenAI plan for higher rate limits

4. **Memory errors**
   - Try processing smaller documents
   - Reduce chunk size in config.py
   - Restart the application

### Debug Mode

For debugging, check the console output when running:
```bash
streamlit run streamlit_app.py --logger.level debug
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review the console logs
3. Ensure your OpenAI API key is valid
4. Try with a smaller PDF first

---

Built with ❤️ using Streamlit, LangChain, and OpenAI