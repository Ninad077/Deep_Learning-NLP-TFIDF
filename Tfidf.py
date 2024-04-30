from sklearn.feature_extraction.text import TfidfVectorizer

sentences = [
    "The movie was good",
    "The movie was boring",
    "The first half was good, the second half was boring"
]

vector = TfidfVectorizer()

tfidf = vector.fit_transform(sentences)
print("words\n\n", vector.get_feature_names_out(),"\n\n")
print("TF-IDF Output\n\n", tfidf.toarray())