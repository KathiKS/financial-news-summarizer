from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk

nltk.download('punkt')

text = open("sample.txt", "r", encoding="utf-8").read()
sentences = nltk.sent_tokenize(text)

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(sentences)
scores = cosine_similarity(tfidf, tfidf).mean(axis=1)
top = scores.argsort()[-2:]
top.sort()
extractive = " ".join([sentences[i] for i in top])

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
output = model.generate(**inputs, max_length=80, min_length=20)
abstractive = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nExtractive Summary:\n", extractive)
print("\nAbstractive Summary:\n", abstractive)
