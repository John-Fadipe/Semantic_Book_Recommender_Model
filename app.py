import os
from flask import Flask, render_template, request
from semantic_book_model import SemanticBookModel

app = Flask(__name__)

book_model = None

@app.before_first_request
def load_model():
    """Load the model only when the first request comes in."""
    global book_model
    if book_model is None:
        print("Loading SemanticBookModel...")
        book_model = SemanticBookModel("csv_folder/books_with_emotions.csv")
        book_model.load_text("text_folder/tagged_description.txt")
        print("Model loaded successfully.")

@app.route("/")
def home():
    return render_template("index.html", book_model=book_model)

@app.route("/recommend", methods=["POST"])
def recommend():
    global book_model

    if book_model is None:
        return render_template(
            "index.html",
            results=[],
            error="Model is still loading. Please try again in a few seconds.",
            book_model=book_model,
        )

    query = request.form.get("query")
    category = request.form.get("category")
    tone = request.form.get("tone")

    if not query:
        return render_template(
            "index.html",
            results=[],
            error="Please enter a query.",
            book_model=book_model,
        )

    results = book_model.recommend_books(query, category, tone)
    return render_template(
        "index.html",
        results=results,
        query=query,
        category=category,
        tone=tone,
        book_model=book_model,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)