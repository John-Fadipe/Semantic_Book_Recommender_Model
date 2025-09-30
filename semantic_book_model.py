import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

class SemanticBookModel:
    def __init__(self, books_path="csv_folder/books_with_emotions.csv"):
        self.db_books = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.raw_df = pd.read_csv(books_path, encoding = "utf-8")
        self.raw_df["large_thumbnail"] = (
            self.raw_df["thumbnail"].fillna("Images/cover-not-found.jpg") + "&fife=w800"
        )

        self.categories = ["All"] + sorted(self.raw_df["simple_categories"].unique())
        self.tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

    def thumbnail_url(self, book_id: str) -> str:
        result = self.raw_df.loc[self.raw_df["book_id"] == book_id, "large_thumbnail"]
        if not result.empty:
            return result.iloc[0]
        return "Images/cover-not-found.jpg"
    
    def load_text(self, path = "text_folder/tagged_description.txt"):
        documents = [
            Document(
            page_content=row["description"],
            metadata={"isbn13": row["isbn13"]}
            )
        for _, row in self.raw_df.iterrows()
        if pd.notna(row["description"])
          ]
        self.db_books = Chroma.from_documents(documents, self.embeddings)

    def retrieve_semantic_recommendations(
        self,
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 10,
    ) -> pd.DataFrame:
        if self.db_books is None:
            raise RuntimeError("Database not loaded. Call load_text() first.")

        recs = self.db_books.similarity_search_with_score(query, k=initial_top_k)

        books_list = [
            rec[0].metadata.get("isbn13") for rec in recs if rec[0].metadata and "isbn13" in rec[0].metadata
        ]
        book_recs = self.raw_df[self.raw_df["isbn13"].isin(books_list)].head(final_top_k)

        if category and category != "All":
            book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)

        tone_sort_map = {
            "Happy": "joy",
            "Surprising": "surprise",
            "Angry": "anger",
            "Suspenseful": "fear",
            "Sad": "sadness",
            "Disgust": "disgust",
        }
        if tone in tone_sort_map:
            book_recs = book_recs.sort_values(by=tone_sort_map[tone], ascending=False).head(final_top_k)

        return book_recs

    def recommend_books(self, query: str, category: str = "All", tone: str = None):
        recommendations = self.retrieve_semantic_recommendations(query, category, tone)
        results = []

        for idx, row in recommendations.iterrows():
            description = row["description"]

            truncated_desc_split = description.split()
            preview = " ".join(truncated_desc_split[:25]) + ("..." if len(truncated_desc_split) > 25 else "")

            authors_split = row["authors"].split(";")
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = row["authors"]

            amazon_link = f"https://www.amazon.com/s?k={row['title'].replace(' ', '+')}"

            results.append({
                "id": idx,
                "thumb": row["large_thumbnail"],
                "title": row["title"],
                "authors": authors_str,
                "preview": preview,
                "full_description": description,
                "link": amazon_link
            })

        return results