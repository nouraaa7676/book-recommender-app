
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD DATA ----------------
books = pd.read_excel("DetailedBooksExcel Cleaned (RemovedBlank).xlsx")
books = books.dropna(subset=['Book Title', 'Author', 'Genre'])

books['Book Title'] = books['Book Title'].astype(str)
books['Author'] = books['Author'].astype(str)
books['Genre'] = books['Genre'].astype(str).str.lower()

books['combined'] = books['Book Title'] + " " + books['Author'] + " " + books['Genre']

# ---------------- VECTORIZER ----------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ---------------- FUNCTIONS ----------------
def recommend_by_genre(genre):
    filtered = books[books['Genre'].str.contains(genre.lower(), na=False)]
    if filtered.empty:
        return ["No books found"]
    return filtered.sample(min(5, len(filtered)))['Book Title'].tolist()

def recommend_by_title(title):
    if title not in books['Book Title'].values:
        return ["Book not found"]

    idx = books[books['Book Title'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return books['Book Title'].iloc[[i[0] for i in scores]].tolist()

def vibe_search(text):
    filtered = books[books['combined'].str.lower().str.contains(text.lower())]
    if filtered.empty:
        return recommend_by_genre(text)
    return filtered.sample(min(5, len(filtered)))['Book Title'].tolist()

# ---------------- FAQ ----------------
faq = {
    "location": "Dubai Digital Park, Silicon Oasis Building A3",
    "delivery": "Free delivery above AED 180",
    "sell books": "Yes, you can sell books with us",
    "order": "Provide order number to track it"
}

def faq_bot(q):
    q = q.lower()
    for k, v in faq.items():
        if k in q:
            return v
    return "Sorry, I don't know the answer."

# ---------------- UI ----------------
st.title("📚 AI Book Recommender + FAQ Chatbot")

menu = st.sidebar.selectbox("Choose", ["Recommender", "FAQ", "Dashboard"])

# ---------------- RECOMMENDER ----------------
if menu == "Recommender":
    option = st.selectbox("Type", ["Genre", "Title", "Vibe"])

    if option == "Genre":
        g = st.text_input("Genre")
        if st.button("Recommend"):
            st.write(recommend_by_genre(g))

    elif option == "Title":
        t = st.text_input("Book Title")
        if st.button("Recommend"):
            st.write(recommend_by_title(t))

    else:
        v = st.text_input("Vibe")
        if st.button("Search"):
            st.write(vibe_search(v))

# ---------------- FAQ ----------------
elif menu == "FAQ":
    q = st.text_input("Ask question")
    if st.button("Ask"):
        st.success(faq_bot(q))

# ---------------- DASHBOARD ----------------
else:
    st.subheader("📊 Insights")
    st.bar_chart(books['Genre'].value_counts())

