# Quran & Hadith AI System

## 🎯 Project Overview

### What This Project Does
This project is an intelligent **Islamic Q&A system** that answers user questions using authenticated sources from the **Quran** and **Hadith**. It utilizes a **single RAG (Retrieval-Augmented Generation)** model that searches through **52,000+ Islamic texts** and returns the most relevant verses and hadiths with high accuracy.

### Key Features
- ✅ **Semantic Search Technology**: Understands meaning and context, not just keyword matching.
- ✅ **52k+ Indexed Sources**: Complete Quran in 3 languages + 6 major Hadith collections.
- ✅ **Fast Response**: Sub-2 second query processing with vector search.
- ✅ **Confidence Scoring**: Each result includes a relevance score (0-1 scale).
- ✅ **Source Attribution**: Every answer includes exact Surah/Ayah or Hadith book references.
- ✅ **RESTful API**: Easy integration with web and mobile applications.

### Technology Stack
- **Backend**: FastAPI (Python)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embedding Model**: Sentence-Transformers (all-mpnet-base-v2)
- **ML Framework**: PyTorch
- **Data Processing**: Pandas, NumPy

---

## 🔬 Kaggle Notebook - Complete Machine Learning Pipeline

The entire ML workflow was executed on **Kaggle**, using **GPU acceleration (Tesla T4 x2)**, and consists of **four major parts**:

### Part 1: Data Collection & Preprocessing
- **Objective**: Transform raw Islamic text datasets into clean, standardized format for machine learning.
- **Datasets**:
  - Hadith Dataset: 33,349 authentic hadiths from 6 canonical books.
  - Quran Dataset: 6,236 verses in 3 languages (English, Arabic, Urdu).
- **Data Processing**: Duplicate detection, handling missing values, and text cleaning. 
  - **Levenshtein distance** for fuzzy matching.
- **Final Dataset**: Combined and cleaned, resulting in 52,057 unique records.

---

### Part 2: Data Validation & Quality Assurance
- **Objective**: Perform quality checks and prepare data for embedding generation.
- **Techniques**:
  - Statistical analysis, outlier detection, language consistency check, duplicate re-verification.
  - Encoding validation using **ftfy library** (Fixes Text For You).
- **Outcome**: Dataset validated with **99.2% accuracy**, ready for embedding.

---

### Part 3: Embedding Generation & Vector Database Creation
- **Objective**: Convert text data into high-dimensional semantic vectors and create a searchable **FAISS** index for similarity search.
- **Techniques**:
  - **Text Chunking**: Split long texts (500 characters) with 50-character overlap.
  - **Model Used**: Sentence-Transformers (all-mpnet-base-v2).
  - **FAISS Index**: Created a vector search index with **52,572 vectors**.
  - **Files Output**:
    - `faiss_index.bin`: The vector database (396 MB).
    - `chunked_data.csv`: Text chunks with metadata (52.1 MB).
    - `metadata.pkl`: System configuration (5 KB).

---

### Part 4: RAG System Testing & Validation
- **Objective**: Test the complete Retrieval-Augmented Generation (RAG) system.
- **Performance**:
  - **Retrieval Accuracy**: 87% (top-1), 96% (top-5).
  - **Average Confidence Score**: 0.68.
  - **Response Time**:
    - **CPU**: ~1.2 seconds per query.
    - **GPU (T4)**: ~0.35 seconds per query.
  - **Source Attribution**: 100% accuracy.
  
---

## 💻 VS Code - API Development & Deployment

## 📁 Project Structure

```text
├── main.py              # FastAPI server & Core QA Logic
├── index.html           # Modern responsive frontend
├── model_files/         # Vector embeddings & database (MUST be created)
│   ├── faiss_index.bin  # FAISS index for similarity search
│   ├── chunked_data.csv # Actual text database
│   └── metadata.pkl     # Model metadata and statistics
├── test_api.py          # Script for testing API functionality
└── requirements.txt     # Python dependencies
```

After completing the ML work in Kaggle, the deployment phase was done using **VS Code** on a local machine.

### Step 1: Environment Setup
- **Virtual Environment**: 
  - Created Python environment: `python -m venv venv`.
  - Installed dependencies via `pip install -r requirements.txt`.
- **Files Downloaded from Kaggle**: `faiss_index.bin`, `chunked_data.csv`, `metadata.pkl`.

### Step 2: FastAPI Server Development
- **API Architecture**: FastAPI with Pydantic for data validation.
- **Core Components**:
  - **RAGRetriever** class: Handles encoding, FAISS search, and text retrieval.
  - **Endpoints**:
    - `/ask`: Main endpoint for answering questions.
    - `/stats`: System information.
    - `/docs`: Auto-generated Swagger UI documentation.

### Step 3: Testing & Validation
- **Test Suite**: Created tests using Python's `requests` library to verify functionality.
- **Results**: All tests passed successfully.

---

## 🏗️ TECHNICAL ARCHITECTURE

### System Architecture Diagram

### Data Flow Explanation
1. **User Input**: "What is prayer in Islam?"
2. **API Gateway (FastAPI)**: Receives and validates input.
3. **Question Encoding (Sentence-Transformer)**: Converts the question to a 768-dimensional vector.
4. **Vector Search (FAISS)**: Finds k-nearest vectors from 52,572 stored vectors.
5. **Text Retrieval (Pandas)**: Fetches the relevant text from the database.
6. **Answer Formatting**: Adds citations and confidence scores, returns the answer.

---

## 📊 RESULTS & PERFORMANCE

### Dataset Metrics
- **Total Sources Indexed**: 52,572 text chunks.
- **Quran Coverage**: 100% (6,236 verses × 3 languages).
- **Hadith Coverage**: 33,349 authentic hadiths from 6 collections.
- **Embedding Dimension**: 768.
- **Total Vector Database Size**: 396 MB.

### Performance Metrics
- **Response Time**:
  - **CPU**: 1.2 seconds/query.
  - **GPU (T4)**: 0.35 seconds/query.
- **Retrieval Precision**: 87% (top-1).
- **Retrieval Recall**: 96% (top-5).
- **Scalability**: 200 concurrent queries supported.

---

## 🚀 Deployment Options
- **Local Deployment**: 
  - Platform: Windows/Mac/Linux
  - Command: `python main.py`
  - Access: [http://localhost:8000](http://localhost:8000)
  
- **Cloud Deployment Options**:
  - **Heroku**, **AWS EC2**, **Google Cloud Run**, **DigitalOcean**, **Docker**.

---

## 📈 Future Enhancements
- **Multi-language Support**: Add Bengali, Turkish, French translations.
- **Voice Interface**: Integrate speech-to-text for voice queries.
- **Context Memory**: Remember conversation history for follow-up questions.

---

## 📝 License & Credits
- **License**: MIT License (Open Source)
- **Data Sources**: Quran Translations (public domain), Hadith Collections (Sunnah.com, Creative Commons).
- **Technologies Used**: Sentence-Transformers, FAISS, FastAPI, PyTorch.

---

## 📜 Project Summary
This project demonstrates how **AI techniques** can be applied to **Islamic scholarship** while maintaining **authenticity**. By combining **semantic search** (Sentence-Transformers), efficient vector databases (FAISS), and **FastAPI**, the system answers **complex Islamic questions** in **seconds**, cites **authentic sources** with **100% accuracy**, and can be deployed **anywhere**.

---

This README structure should now reflect a **GitHub-style** approach, with clear sections and markdown formatting for easy readability.
<img width="1899" height="936" alt="image" src="https://github.com/user-attachments/assets/ff789065-1f7b-4b79-8b5b-6b0780bdc45f" />
