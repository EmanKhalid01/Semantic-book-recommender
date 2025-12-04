import gradio as gr
import pandas as pd
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load Sentence transformers
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load chroma collection
PERSIST_DIR = r"D:\book-recommender\chroma_db"
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_collection("books_collection")

# Load classification pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli"
)

# Load sentiment model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Load dataset
books = pd.read_csv("./Notebooks/books_classified.csv")

def search_books(query, n=5):
    """Semantic book search using ChromaDB"""
    results = collection.query(
        query_texts=[query],
        n_results=n
    )
    if len(results["ids"]) == 0 or len(results["ids"][0]) == 0:
        return [{"Error": "No similar books found."}]

    titles = [meta["title"] for meta in results["metadatas"][0]]
    descs = results["documents"][0]
    ids = results["ids"][0]

    output = []


    for t, d, i in zip(titles, descs, ids):
        row = books[books["isbn13"] == int(i)]
        category = row["predicted_category"].values[0]
        senti = row["sentiment"].values[0] if "sentiment" in books.columns else "Not Available"

        output.append({
            "Title": t,
            "Description": d[:300] + "...",
            "Category": category,
            "Sentiment": senti
        })

    return output
 
def classify_text(text):
    """Classify text into Fiction / Non-fiction"""
    labels = ["Fiction", "Non-fiction"]
    result = classifier(text, candidate_labels=labels)
    return {
        "Predicted Category": result["labels"][0],
        "Confidence": round(result["scores"][0], 3)
    }

def sentiment(text):
    """Sentiment analysis (Positive/ Negative/ Neutral)"""
    result = sentiment_analyzer(text)[0]
    return {
        "Sentiment": result["label"],
        "Confidence": round(result["score"], 3)
    }

# Gradio UI Layout
 
# Gradio UI Layout
 
theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="gray",
).set(
    body_background_fill="#F2F2F2",       # Light gray background
    button_primary_background_fill="#1A2A6C",  # Navy blue
    button_primary_background_fill_hover="#162359",
    button_secondary_background_fill="#1A2A6C",
    button_secondary_background_fill_hover="#162359",
    button_primary_text_color="#F5F5F5",        # Off-white text
    button_secondary_text_color="#F5F5F5",
    block_background_fill="#FFFFFF",      # White cards
    block_shadow="0px 2px 8px rgba(0,0,0,0.12)",
    block_border_width="0px",
    block_padding="20px",
)

with gr.Blocks() as demo:

    gr.Markdown(
        """
        <div style="text-align: center;  background-color: #1A2A6C;
                    padding: 25px; border-radius: 12px;">
            <h1 style="color: #f5f5f5; font-size: 36px; font-weight: 600; margin: 0;">
                       📚 AI Book Recommender Dashboard</h1>
            <p style="font-size: 18px; color: #EAEAEA; margin-top: 8px;">
            • Smart Search&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            • Category Prediction&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            • Sentiment Analysis
           </p>
        </div>
        """
    ) 

    with gr.Row(equal_height=True):

     # ----- Box 1: Book Search -----
        with gr.Column():
            gr.Markdown("### 🔍 Book Search (Semantic + Metadata)")
            query = gr.Textbox(
                label="Enter a topic or description",
                placeholder="e.g., love, war, adventure, machine learning...",
            )
            btn1 = gr.Button("🔎 Search Books", variant="secondary")
            output1 = gr.JSON(label="Top Matches")

        # ----- Box 2: Category Prediction -----
        with gr.Column():
            gr.Markdown("### 📖 Fiction / Nonfiction Classifier")
            text_input = gr.Textbox(
                label="Enter book description",
                placeholder="Paste book blurb or plot summary...",
            )
            btn2 = gr.Button("📘 Predict Category", variant="secondary")
            output2 = gr.JSON(label="Category Prediction")

        # ----- Box 3: Sentiment Analysis -----
        with gr.Column():
            gr.Markdown("### 😊 Sentiment Analyzer")
            text_sent = gr.Textbox(
                label="Enter text",
                placeholder="Paste a review or any text...",
            )
            btn3 = gr.Button("💬 Analyze Sentiment", variant="secondary")
            output3 = gr.JSON(label="Sentiment Result")

    # Button Actions
    btn1.click(search_books, inputs=query, outputs=output1)
    btn2.click(classify_text, inputs=text_input, outputs=output2)
    btn3.click(sentiment, inputs=text_sent, outputs=output3)

demo.launch(share=True, inbrowser=True, theme=theme)   