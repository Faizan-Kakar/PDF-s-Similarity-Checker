# 🧠 PDF Document Similarity Checker

This project is an AI/ML-powered solution designed to compare the similarity between two PDF documents, typically used for academic plagiarism detection or research content comparison. It provides both a **quantitative similarity score** and a **qualitative summary** of overlapping content using state-of-the-art language models.

---

## 🔍 Project Overview

Given two PDF documents, the system:

1. Extracts and preprocesses the text.
2. Chunks the text and generates sentence embeddings using a transformer model.
3. Computes similarity using cosine similarity between vector representations.
4. Identifies overlapping chunks.
5. Summarizes those chunks using a T5-based transformer summarizer.
6. Outputs a user-friendly similarity report with actionable suggestions.

---

## 🧰 Tech Stack & Libraries Used

| Technology | Purpose |
|------------|---------|
| **Python** | Programming Language |
| `fitz (PyMuPDF)` | PDF text extraction |
| `nltk` | Tokenization, stopword removal, lemmatization |
| `sentence-transformers` | Sentence embeddings (`all-mpnet-base-v2`) |
| `transformers` (Hugging Face) | Summarization using `t5-small` |
| `scikit-learn` | Cosine similarity computation |
| `NumPy` | Numerical operations |

---


## 🚀 How to Use

### 🧪 Option 1:  Run Notebook in google colab (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdf_similarity_checker.git
   

2. Launch the Jupyter Notebook
   ```bash
   jupyter notebook notebook/Document_Similarity_Detection.ipynb
   
3. Upload your PDFs and run the cells in sequence to get results.


---

### 🧪 Option 2: Run via Python Scripts

If you prefer using Python scripts directly, follow the structured flow below.


#### 🚀 Steps to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pdf_similarity_checker.git
   cd pdf_similarity_checker
   
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   
3. **Run the Script**:
- Place the two PDF files you want to compare in the data/ directory and run the script.
- Run the src/pdf_preprocessing.py script
- Run the src/similarity_checker.py
- Last to check the results run the output/result_summary.py


---
  
## 📝 Sample Output

🔍 Similarity Score: 0.8124

✅ Your submitted research is highly similar to an existing paper.




---

## 🤖 Models Used

- **Sentence Embedding**: `all-mpnet-base-v2` from [`sentence-transformers`](https://www.sbert.net/)
- **Summarization**: `t5-small` from [Hugging Face `transformers`](https://huggingface.co/t5-small)

---

## 📌 Future Improvements

- Web-based UI for easy uploads and visualization.
- Batch processing of multiple document comparisons.
- Integration with plagiarism APIs or LMS platforms.

---

## 👤 Author

**Faizan Khan**  
AI/ML Enthusiast | Portfolio Project

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to use and modify it.

