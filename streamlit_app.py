import streamlit as st
import pandas as pd
import numpy as np
import re
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CẤU HÌNH & STATE ---
st.set_page_config(layout="wide", page_title="Women's Beauty & Jewelry", page_icon="💎")

# Khởi tạo Session State
if 'history' not in st.session_state: st.session_state.history = []
if 'favorites' not in st.session_state: st.session_state.favorites = []
if 'view_asin' not in st.session_state: st.session_state.view_asin = None 
# Lưu danh sách "Xu hướng" để không bị reset khi click
if 'trends' not in st.session_state: st.session_state.trends = None

# --- 2. HÀM XỬ LÝ TEXT & ẢNH (NÂNG CẤP) ---
def clean_text_display(text):
    if pd.isna(text): return "Thông tin đang cập nhật..."
    text = str(text)
    if text.startswith("['") and text.endswith("']"): text = text[2:-2]
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    return " ".join(text.split())

def get_hd_image(url):
    """Chuyển link ảnh thumbnail thành HD bằng cách xóa mã resize"""
    if pd.isna(url) or 'http' not in str(url): return "https://via.placeholder.com/300x400?text=No+Image"
    # Xóa các đoạn mã như ._AC_US40_ hoặc ._SX300_ để lấy ảnh gốc
    hd_url = re.sub(r'\._[A-Z]{2}\d+(,_\d+)?_(\.[a-z]+)$', r'\2', str(url))
    hd_url = re.sub(r'\._AC_.*(\.[a-z]+)$', r'\1', hd_url)
    return hd_url

# --- 3. LOAD DỮ LIỆU ---
@st.cache_resource
def load_data_and_model():
    input_file = 'Women_Cosmetics_Jewelry_Clean.csv'
    try:
        df = pd.read_csv(input_file)
    except:
        st.error(f"Chưa có file '{input_file}'. Hãy chạy Preprocessing trước!")
        st.stop()
        
    df = df.drop_duplicates(subset=['asin']).copy()
    
    # Xử lý dữ liệu hiển thị
    df['clean_desc'] = df['item_text'].apply(clean_text_display)
    df['clean_title'] = df['title'].apply(lambda x: html.unescape(str(x)))
    df['hd_image'] = df['image_url_clean'].apply(get_hd_image) # Tạo cột ảnh HD
    if 'price_numeric' not in df.columns: df['price_numeric'] = 0.0
    
    # Model Content-based
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df['item_text'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, cosine_sim

df, cosine_sim = load_data_and_model()

# Khởi tạo danh sách Xu hướng (chỉ chạy 1 lần đầu)
if st.session_state.trends is None:
    st.session_state.trends = df.sample(min(12, len(df)))

# --- 4. LOGIC CHỨC NĂNG ---
def update_history(item):
    # Xóa cũ thêm mới lên đầu (Không lặp)
    st.session_state.history = [h for h in st.session_state.history if h['asin'] != item['asin']]
    st.session_state.history.insert(0, {'asin': item['asin'], 'title': item['clean_title']})
    st.session_state.history = st.session_state.history[:15]

def toggle_favorite(asin):
    if asin in st.session_state.favorites:
        st.session_state.favorites.remove(asin)
    else:
        st.session_state.favorites.append(asin)

def view_product(asin):
    st.session_state.view_asin = asin

def go_home():
    st.session_state.view_asin = None

def get_recs(asin, top_k=5):
    indices = pd.Series(df.index, index=df['asin'])
    if asin not in indices: return pd.DataFrame()
    idx = indices[asin]
    if idx >= cosine_sim.shape[0]: return pd.DataFrame()
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_k+1]
    return df.iloc[[i[0] for i in sim_scores if i[0] < len(df)]]

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🎀 Menu")
    if st.button("🏠 Trang chủ", use_container_width=True, type="primary"):
        go_home()
        st.rerun()
        
    tab1, tab2 = st.tabs(["❤️ Yêu thích", "🕒 Lịch sử"])
    with tab1:
        if st.session_state.favorites:
            # Lọc danh sách yêu thích từ dataframe để lấy thông tin ảnh/tên
            fav_items = df[df['asin'].isin(st.session_state.favorites)]
            for _, item in fav_items.iterrows():
                with st.container(border=True):
                    c_img, c_info = st.columns([1, 2])
                    c_img.image(item['hd_image'])
                    c_info.caption(item['clean_title'][:40])
                    # Nút xem
                    if c_info.button("Xem", key=f"fav_view_{item['asin']}"):
                        view_product(item['asin'])
                        st.rerun()
                    # Nút xóa
                    if c_info.button("Xóa", key=f"fav_del_{item['asin']}"):
                        toggle_favorite(item['asin'])
                        st.rerun()
        else: st.info("Chưa có sản phẩm yêu thích")
        
    with tab2:
        if st.session_state.history:
            for h in st.session_state.history:
                if st.button(f"👁️ {h['title'][:25]}...", key=f"hist_{h['asin']}", use_container_width=True):
                    view_product(h['asin'])
                    st.rerun()
            if st.button("🗑️ Xóa lịch sử"):
                st.session_state.history = []
                st.rerun()

# --- 6. GIAO DIỆN CHÍNH ---
st.title("💎 Women's Cosmetics & Jewelry Store")

# Thanh tìm kiếm
search_options = df['clean_title'].tolist()
selected = st.selectbox("🔍 Tìm kiếm sản phẩm:", [""] + search_options, index=0)
if selected:
    found_asin = df[df['clean_title'] == selected].iloc[0]['asin']
    if found_asin != st.session_state.view_asin:
        view_product(found_asin)
        st.rerun()

st.divider()

# --- TRANG CHI TIẾT SẢN PHẨM ---
if st.session_state.view_asin:
    try:
        item = df[df['asin'] == st.session_state.view_asin].iloc[0]
        update_history(item)
        
        # Nút Back
        if st.button("⬅️ Quay lại trang chủ"):
            go_home()
            st.rerun()

        # Layout Thông tin
        c1, c2 = st.columns([1, 1.5])
        with c1:
            # ẢNH HD Ở ĐÂY
            st.image(item['hd_image'], width=500) 
        
        with c2:
            st.header(item['clean_title'])
            st.markdown(f"🏷️ **Thương hiệu:** {item['brand']}")
            st.subheader(f"💵 Giá: :red[${item['price_numeric']:.2f}]")
            
            # Logic nút Yêu thích
            is_fav = item['asin'] in st.session_state.favorites
            btn_label = "❤️ Bỏ thích" if is_fav else "🤍 Yêu thích"
            btn_type = "primary" if is_fav else "secondary"
            
            if st.button(btn_label, type=btn_type, key="main_fav_btn"):
                toggle_favorite(item['asin'])
                st.rerun()

            # Mô tả sạch (đã tách riêng clean_desc)
            with st.container(border=True):
                st.markdown("**📝 Mô tả chi tiết:**")
                st.write(item['clean_desc'])
        
        st.divider()
        st.subheader("✨ Gợi ý sản phẩm tương tự")
        recs = get_recs(item['asin'])
        
        # Grid 5 cột
        cols = st.columns(5)
        for i, (_, r) in enumerate(recs.iterrows()):
            with cols[i]:
                with st.container(border=True):
                    # Ảnh HD trong gợi ý
                    st.image(r['hd_image'], use_container_width=True) 
                    st.caption(f"{r['clean_title'][:40]}...")
                    st.write(f":red[${r['price_numeric']:.2f}]")
                    if st.button("Xem ngay", key=f"rec_{r['asin']}"):
                        view_product(r['asin'])
                        st.rerun()

    except Exception as e:
        st.error("Sản phẩm không tồn tại hoặc đã bị lọc.")
        st.write(e)
        if st.button("Về trang chủ"):
            go_home()
            st.rerun()

# --- TRANG CHỦ (XU HƯỚNG) ---
else:
    st.subheader("🔥 Xu hướng & Gợi ý hôm nay")
    st.caption("Các sản phẩm hot nhất được lựa chọn ngẫu nhiên cho bạn.")
    
    # Grid 4 cột cho đẹp
    cols = st.columns(4)
    # Lấy data từ session_state để không bị reset khi click
    for i, (_, r) in enumerate(st.session_state.trends.iterrows()):
        with cols[i % 4]:
            with st.container(border=True):
                # Ảnh HD
                st.image(r['hd_image'], use_container_width=True)
                st.markdown(f"**{r['clean_title'][:50]}...**")
                st.write(f"💰 :red[${r['price_numeric']:.2f}]")
                
                # Nút xem chi tiết
                if st.button("Xem chi tiết", key=f"trend_{r['asin']}", use_container_width=True):
                    view_product(r['asin'])
                    st.rerun()