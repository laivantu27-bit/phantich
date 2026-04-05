import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

stop_words = ["là", "và", "có", "rất", "thì", "một", "những", "các", "ở", "không", "của", "cho", "với"]

def preprocess_text(text):
    """Hàm làm sạch văn bản"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\d+', '', text)     

    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

file_path = "ITA105_Lab_4_Hotel_reviews.csv"

print(f"Đang đọc dữ liệu từ: {file_path}...")
df = pd.read_csv(file_path)

text_col = df.columns[0] 
print("Đang tiến hành làm sạch dữ liệu...")
df['Cleaned_Text'] = df[text_col].apply(preprocess_text)
print("\n--- Dữ liệu sau khi tiền xử lý (5 dòng đầu) ---")
print(df[['Cleaned_Text']].head())
print("\n" + "="*40)
print("PHẦN 1: TRIỂN KHAI TF-IDF")
print("="*40)
tfidf_vectorizer = TfidfVectorizer()

X_tfidf = tfidf_vectorizer.fit_transform(df['Cleaned_Text'])
print(f"Kích thước ma trận TF-IDF: {X_tfidf.shape} (Số lượng câu x Số lượng từ vựng)")
print("Danh sách 10 từ vựng đầu tiên trong từ điển TF-IDF:")
print(tfidf_vectorizer.get_feature_names_out()[:10])
print("\n" + "="*40)
print("PHẦN 2: TRIỂN KHAI WORD2VEC")
print("="*40)

sentences = [text.split() for text in df['Cleaned_Text']]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
print("Đã huấn luyện xong mô hình Word2Vec.")
sample_word = sentences[0][0] if len(sentences[0]) > 0 else None

if sample_word and sample_word in w2v_model.wv:
    word_vector = w2v_model.wv[sample_word]
    print(f"\nVector biểu diễn của từ '{sample_word}' có {len(word_vector)} chiều.")
    print("5 giá trị đầu tiên của vector này là:")
    print(word_vector[:5])
else:
    print("\nKhông tìm thấy từ mẫu trong tập từ vựng Word2Vec.")
    