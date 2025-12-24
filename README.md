# 📖 Quran & Hadith AI - Authentic Islamic Q&A

**Turjoy's Quran & Hadith AI** is a powerful, authentic Islamic Question & Answering system. It leverages state-of-the-art semantic search technology to provide accurate answers directly sourced from over 50,000+ Quran verses and Hadiths, ensuring reliable knowledge for seekers of truth.

---

## 🚀 Key Features

- **🎯 Semantic Search**: Finds relevant verses and Hadiths even if your keywords don't match exactly, thanks to Sentence-Transformers.
- **📚 Vast Database**: Access to the Holy Quran, 6 major Hadith books (Kutub al-Sittah), and other trusted reference sources.
- **🔗 Source Referencing**: Every answer is backed by direct references (Surah/Ayat or Hadith numbers) with high-confidence matching.
- **✨ Professional UI**: A sleek, modern, and responsive dashboard built with a refined emerald and gold aesthetic.
- **⚡ Fast API**: Powered by FastAPI and FAISS for sub-second retrieval times.
- **📊 System Stats**: Built-in monitoring to track the total number of sources and languages supported.

---

## 🛠️ Technology Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
- **AI Model**: [Sentence-Transformers](https://www.sbert.net/) (for high-quality embeddings)
- **Frontend**: Vanilla HTML5, CSS3 (Modern UI with Glassmorphism), and JavaScript (Fetch API)
- **Data Handling**: Pandas, NumPy, Pickle

---

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

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```powershell
git clone <repository-url>
cd Quran_hadith_api
```

### 2. Set up Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Prepare Model Files
Ensure you have the `model_files` folder in the root directory with the following files:
- `faiss_index.bin`
- `chunked_data.csv`
- `metadata.pkl`

### 5. Run the Application
```powershell
uvicorn main:app --reload
```
The app will be available at `http://localhost:8000`.

---

## 🧪 API Documentation

The API automatically generates interactive documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Primary Endpoints:
- `POST /ask`: Ask a single question about Islam.
- `POST /batch_ask`: Submit multiple questions at once.
- `GET /stats`: Retrieve system statistics (total verses/hadiths).
- `GET /search`: Perform a raw search across the database.

---

## ⚖️ Disclaimer
This AI system provides information based on processed datasets of Quran and Hadith. Users are encouraged to cross-reference with scholars and original texts for religious rulings (Fatwas).

---
© 2025 Turjoy's Quran & Hadith Training
