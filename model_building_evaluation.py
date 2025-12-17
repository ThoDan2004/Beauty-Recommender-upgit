import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# --- CẤU HÌNH ---
pd.set_option('display.max_colwidth', 50)
INPUT_FILE = 'Women_Cosmetics_Jewelry_Clean.csv'
RATING_COLUMN = 'overall'

# --- 1. TẢI DỮ LIỆU ---
print("--- 1. Tải và chuẩn bị dữ liệu mô hình ---")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"❌ LỖI: Không tìm thấy file {INPUT_FILE}. Hãy chạy file Preprocessing trước!")
    exit()

# Lọc bỏ dòng lỗi nếu có (dù file 1 đã lọc kỹ)
df_model = df.dropna(subset=[RATING_COLUMN])
print(f"📊 Tổng số Items sạch: {df_model['asin'].nunique()}")

# Lọc tương tác: Chỉ lấy user có ít nhất 2 đánh giá để Collaborative Filtering tốt hơn
# Nếu dữ liệu ít (<3000 item), ta giảm điều kiện này xuống để bảo toàn dữ liệu
min_reviews = 1 if df_model['asin'].nunique() < 3000 else 2
item_counts = df_model['asin'].value_counts()
user_counts = df_model['reviewerID'].value_counts()

df_model = df_model[df_model['asin'].isin(item_counts[item_counts >= min_reviews].index)]
df_model = df_model[df_model['reviewerID'].isin(user_counts[user_counts >= min_reviews].index)]

print(f"-> Dữ liệu Training cuối cùng: {len(df_model)} dòng tương tác.")

# --- 2. CONTENT-BASED (Yêu cầu PDF: Vector hóa TF-IDF) ---
print("\n--- 2. Xây dựng Content-based (TF-IDF) ---")
df_items = df_model.drop_duplicates(subset=['asin'])
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
# Sử dụng cột item_text (đã gộp Title + Brand + Description)
tfidf_matrix = tfidf.fit_transform(df_items['item_text'].fillna(''))
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Map ID
id_to_index = {asin: i for i, asin in enumerate(df_items['asin'])}

# --- 3. COLLABORATIVE FILTERING (Yêu cầu PDF: Model Recommendation) ---
print("\n--- 3. Xây dựng Collaborative Filtering (KNN) ---")
rating_matrix = df_model.pivot_table(index='reviewerID', columns='asin', values=RATING_COLUMN)
rating_matrix_sparse = csr_matrix(rating_matrix.fillna(0).values)

knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn_model.fit(rating_matrix_sparse)

# --- 4. ĐÁNH GIÁ MÔ HÌNH (Yêu cầu PDF: RMSE, MAE, Precision, Recall) ---
print("\n--- 4. Đánh giá Mô hình (Evaluation) ---")
train_df, test_df = train_test_split(df_model, test_size=0.2, random_state=42)

user_map = {u: i for i, u in enumerate(rating_matrix.index)}
item_map = {i: k for k, i in enumerate(rating_matrix.columns)}

def predict_rating(user_id, item_id):
    if user_id not in user_map or item_id not in item_map:
        return df_model[RATING_COLUMN].mean()
    u_idx = user_map[user_id]
    i_idx = item_map[item_id]
    distances, indices = knn_model.kneighbors(rating_matrix_sparse[u_idx], n_neighbors=5)
    
    neighbor_ratings = []
    for idx in indices.flatten():
        val = rating_matrix.iloc[idx, i_idx]
        if val > 0: neighbor_ratings.append(val)
    return np.mean(neighbor_ratings) if neighbor_ratings else df_model[RATING_COLUMN].mean()

print("Đang chạy dự đoán (Evaluation)...")
test_df = test_df.copy()
test_df['predicted'] = test_df.apply(lambda x: predict_rating(x['reviewerID'], x['asin']), axis=1)

# Metrics
rmse = np.sqrt(mean_squared_error(test_df[RATING_COLUMN], test_df['predicted']))
mae = mean_absolute_error(test_df[RATING_COLUMN], test_df['predicted'])

threshold = 4.0
k_recs = test_df[test_df['predicted'] >= threshold]
true_pos = k_recs[k_recs[RATING_COLUMN] >= threshold]

precision = len(true_pos) / len(k_recs) if len(k_recs) > 0 else 0
recall = len(true_pos) / len(test_df[test_df[RATING_COLUMN] >= threshold]) if len(test_df) > 0 else 0

print(f"\n=========================================")
print(f"| KẾT QUẢ ĐÁNH GIÁ (Theme: Women/Beauty)|")
print(f"=========================================")
print(f"| RMSE         : {rmse:.4f}             |")
print(f"| MAE          : {mae:.4f}             |")
print(f"| Precision@K  : {precision:.4f}             |")
print(f"| Recall@K     : {recall:.4f}             |")
print(f"=========================================")