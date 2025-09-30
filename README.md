## 📚 Semantic Book Recommender

This project is a semantic search–powered book recommendation app that uses natural language processing (NLP) to recommend books based on user input, such as emotions, themes, or descriptions. It leverages embeddings and vector similarity search to provide meaningful book suggestions with cover thumbnails, author details, and Amazon links.

## 📌 Project Overview

- Recommends books based on user queries (e.g., “hopeful and inspiring stories”).

- Uses sentence embeddings to match queries with book descriptions.

- Includes a Flask web app interface with a modern, responsive frontend.

- Displays recommended books with thumbnails, authors, previews, and expandable descriptions.

- Links directly to Amazon book pages for easy access.

## 🧠 Model

- Embedding Model: sentence-transformers (Hugging Face)

- Vector Store: ChromaDB for similarity search

- Frameworks: LangChain + HuggingFace integrations

- Search: Semantic matching instead of keyword search

## 🚀 App Features

- Frontend: HTML + CSS (custom styles)

- Backend: Flask + Pandas

- Input: Free text query (e.g., “books about resilience and growth”)

- Output: Ranked list of books with cover images and metadata

Expandable book descriptions (“Show Preview” dropdown)