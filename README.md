# 📚 AI Semantic Book Recommender Dashboard
 **AI-powered book recommender system** is an interactive web-based application that enables semantic book search, category classification, and sentiment analysis.<br>
This project is built with Python and leverages modern NLP techniques for recommendation, classification, and sentiment analysis.
# 🚀 Project Overview

  # Key Components:
  - **Dashboard.py**: A Gradio-based web app for interactive book recommendations.
  - **Notebooks/**: Jupyter notebooks for data exploration, text classification, semantic search, and sentiment analysis.
  - **chroma_db/**: Chroma vector database for semantic search.
  - **.venv_Dashboard/** and **.venv_Notebooks/**: Separate Python virtual environments for Dashboard and notebooks.
 
  # Key Features:
  **1.** **Semantic Search**  
   - Search books using vector similarity on book descriptions.
   - Powered by **ChromaDB** and **Sentence Transformers**.

  **2.** **Fiction / Non-Fiction Classifier**  
   - Classifies book descriptions into categories(Fiction, Non-fiction) using a **Zero-Shot Classification pipeline**.

  **3.** **Sentiment Analysis**  
   - Analyze sentiment(Positive, Negative, Neutral) of text, reviews, and summaries etc. using **DistilBERT sentiment model**.

  4. **Interactive Dashboard**  
   - Built with **Gradio**.  
   - Clean, modern UI with HEX color theme.  
   - Supports search, category prediction, and sentiment analysis in one interface.
  
# 🔧 Technologies Used

- **Python 3.11** – Programming Language  
- **Gradio** – Web-based UI framework for interactive dashboards  
- **ChromaDB** – Vector database for semantic search  
- **Sentence Transformers** – For embedding book descriptions  
- **Hugging Face Transformers** – For classification and sentiment analysis  
- **Pandas** – Data manipulation and handling CSV files 

# 🛠 Installation

  1. **Clone this repo:**
    https://github.com/EmanKhalid01/Semantic-book-recommender.git
  2. **Create virtual Environments:**
    - For notebooks
    python -m venv .venv_Notebooks
    - For Dashboard
    python -m venv .venv_Dashboard
  3. **Activate environment & install dependencies:**
    - Notebooks
    .venv_Notebooks\Scripts\activate      # Windows
    pip install -r requirements_notebooks.txt
    - Dashboard
    .venv_Dashboard\Scripts\activate      # Windows
    pip install -r requirements_dashboard.txt

#  ♻ Usage
  **1.** **Run Notebooks:**
   - Open Jupyter:
   - .venv_Notebooks\Scripts\activate
   - Open Jupyter notebook.
   - Select the kernel: .venv_Notebooks
   - Run all notebooks (data-explore.ipynb, vector-search.ipynb, text-classification.ipynb, sentiment-analysis.ipynb)
   **2.** **Launch Dashboard:**
   - .venv_Dashboard\Scripts\activate   # Windows
   - python Dashboard.py
   - Inbrower=true will automatically open the Dashboard in the Browser.
   - Or use share=True in demo.launch() to get a temporary public URL.

# 📂 Folder Structure
book-recommender/
-├── Dashboard.py
-├── chroma_db/
-├── Notebooks/
-├── .venv_Dashboard/
-├── .venv_Notebooks/
-└── .gradio/
-├── requirements_Dashboard.txt
-├── requirements_Notebooks.txt
-└── .gradio/
-├── README.md
-├── .gitignore

# 📂 Notebooks
All exploration and preprocessing notebooks are in the Notebooks/ folder:

- data-explore.ipynb – Dataset exploration
- vector-search.ipynb – Semantic search using embeddings
- text-classification.ipynb – Book classification pipeline
- sentiment-analysis.ipynb – Sentiment model testing

# 📌 Notes
- Ensure CSV files are in Notebooks/ folder.
- ChromaDB stores vectors in chroma_db/.
- Use separate environments to avoid dependency conflicts.

# 🔗 References



# ⚖️ License
This project is licensed under the MIT License. See the LICENSE file for details.

# 👨‍💻 Author
Eman Khalid – Python & AI Enthusiast
- LinkedIn: https://linkedin.com/in/eman-khalid001

