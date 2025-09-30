from flask import Flask, render_template, request
from semantic_book_model import SemanticBookModel

app = Flask(__name__)

book_model = SemanticBookModel("CSV Folder/books_with_emotions.csv")
book_model.load_text("Text Folder/tagged_description.txt") 


@app.route("/")
def home():
    return render_template("index.html", book_model = book_model)


@app.route("/recommend", methods = ["POST"])
def recommend():    
    query = request.form.get("query")
    category = request.form.get("category")
    tone = request.form.get("tone")
    if not query:
        return render_template("index.html",
                               results = [],
                               error = "Please enter a query.",
                               book_model = book_model)

    results = book_model.recommend_books(query, category, tone)
    return render_template("index.html", 
                           results = results, 
                           query = query, 
                           category = category, 
                           tone = tone,
                           book_model = book_model)


if __name__ == "__main__":
    app.run(debug=True)