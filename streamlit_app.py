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
if 'trends' not in st.session_state: st.session_state.trends = None

# --- 2. HÀM XỬ LÝ TEXT & ẢNH ---
def clean_text_display(text):
    if pd.isna(text): return "Thông tin đang cập nhật..."
    text = str(text)
    if text.startswith("['") and text.endswith("']"): text = text[2:-2]
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    return " ".join(text.split())

def get_hd_image(url):
    if pd.isna(url) or 'http' not in str(url): return "https://via.placeholder.com/300x400?text=No+Image"
    # Xóa mã resize để lấy ảnh gốc nét nhất
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
    
    # --- QUAN TRỌNG: RESET INDEX ĐỂ TRÁNH LỖI LỆCH GỢI Ý ---
    df = df.reset_index(drop=True)
    
    # Xử lý hiển thị
    df['clean_desc'] = df['item_text'].apply(clean_text_display)
    df['clean_title'] = df['title'].apply(lambda x: html.unescape(str(x)))
    df['hd_image'] = df['image_url_clean'].apply(get_hd_image)
    if 'price_numeric' not in df.columns: df['price_numeric'] = 0.0
    
    # Model: Tăng cường trọng số cho Title để gợi ý bớt "lạc đề"
    # Gấp 3 lần Title để ép nó tìm món cùng loại
    df['training_text'] = (df['clean_title'] + " " + df['clean_title'] + " " + df['clean_title'] + " " + df['item_text']).fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    tfidf_matrix = tfidf.fit_transform(df['training_text'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, cosine_sim

df, cosine_sim = load_data_and_model()

# Init Trends
if st.session_state.trends is None:
    st.session_state.trends = df.sample(min(12, len(df)))

# --- 4. HÀM CALLBACK (XỬ LÝ SỰ KIỆN NÚT BẤM) ---
# Đây là chìa khóa để sửa lỗi StreamlitAPIException và lỗi phải ấn 2 lần

def cb_view_product(asin):
    """Callback khi ấn xem sản phẩm: Cập nhật view và xóa tìm kiếm"""
    st.session_state.view_asin = asin
    st.session_state.search_box = None # Xóa tìm kiếm an toàn ở đây
    
    # Cập nhật lịch sử
    item = df[df['asin'] == asin].iloc[0]
    st.session_state.history = [h for h in st.session_state.history if h['asin'] != asin]
    st.session_state.history.insert(0, {'asin': asin, 'title': item['clean_title']})
    st.session_state.history = st.session_state.history[:15]

def cb_go_home():
    """Callback về trang chủ"""
    st.session_state.view_asin = None
    st.session_state.search_box = None

def cb_toggle_favorite(asin):
    """Callback thích/bỏ thích"""
    if asin in st.session_state.favorites:
        st.session_state.favorites.remove(asin)
    else:
        st.session_state.favorites.append(asin)

def cb_search():
    """Callback khi gõ tìm kiếm"""
    if st.session_state.search_box:
        found = df[df['clean_title'] == st.session_state.search_box]
        if not found.empty:
            cb_view_product(found.iloc[0]['asin'])

def get_recs(asin, top_k=5):
    try:
        # Lấy index chính xác nhờ đã reset_index ở bước load
        idx = df.index[df['asin'] == asin].tolist()[0]
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Bỏ qua index 0 (chính nó)
        sim_scores = sim_scores[1:top_k+1]
        
        item_indices = [i[0] for i in sim_scores]
        return df.iloc[item_indices]
    except:
        return pd.DataFrame()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🎀 Menu")
    # Dùng on_click thay vì if st.button
    st.button("🏠 Trang chủ", use_container_width=True, type="primary", on_click=cb_go_home)
        
    tab1, tab2 = st.tabs(["❤️ Yêu thích", "🕒 Lịch sử"])
    with tab1:
        if st.session_state.favorites:
            fav_items = df[df['asin'].isin(st.session_state.favorites)]
            for _, item in fav_items.iterrows():
                with st.container(border=True):
                    c_img, c_info = st.columns([1, 2])
                    c_img.image(item['hd_image'])
                    c_info.caption(item['clean_title'][:40])
                    # Nút Xem dùng Callback
                    c_info.button("Xem", key=f"fav_v_{item['asin']}", on_click=cb_view_product, args=(item['asin'],))
                    # Nút Xóa dùng Callback
                    c_info.button("Xóa", key=f"fav_d_{item['asin']}", on_click=cb_toggle_favorite, args=(item['asin'],))
        else: st.info("Trống")
        
    with tab2:
        if st.session_state.history:
            for h in st.session_state.history:
                st.button(f"👁️ {h['title'][:25]}...", key=f"hist_{h['asin']}", 
                         use_container_width=True, 
                         on_click=cb_view_product, args=(h['asin'],))
            st.button("🗑️ Xóa lịch sử", on_click=lambda: st.session_state.update(history=[]))

# --- 6. GIAO DIỆN CHÍNH ---
st.title("💎 Women's Cosmetics & Jewelry Store")

# Thanh tìm kiếm (Có callback on_change)
st.selectbox(
    "🔍 Tìm kiếm sản phẩm:", 
    options=df['clean_title'].tolist(), 
    index=None, 
    key="search_box", 
    placeholder="Nhập tên sản phẩm...",
    on_change=cb_search # Chạy hàm này ngay khi enter
)

st.divider()

# --- TRANG CHI TIẾT ---
if st.session_state.view_asin:
    try:
        # Lấy lại item từ ASIN đang view
        item = df[df['asin'] == st.session_state.view_asin].iloc[0]
        
        st.button("⬅️ Quay lại trang chủ", on_click=cb_go_home)

        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.image(item['hd_image'], width=500) 
        
        with c2:
            st.header(item['clean_title'])
            st.markdown(f"🏷️ **Thương hiệu:** {item['brand']}")
            st.subheader(f"💵 Giá: :red[${item['price_numeric']:.2f}]")
            
            # Nút yêu thích (Callback)
            is_fav = item['asin'] in st.session_state.favorites
            btn_label = "❤️ Bỏ thích" if is_fav else "🤍 Yêu thích"
            btn_type = "primary" if is_fav else "secondary"
            st.button(btn_label, type=btn_type, key="main_fav_btn", 
                     on_click=cb_toggle_favorite, args=(item['asin'],))

            with st.container(border=True):
                st.markdown("**📝 Mô tả chi tiết:**")
                st.write(item.get('clean_desc', item['item_text']))
        
        st.divider()
        st.subheader("✨ Gợi ý sản phẩm tương tự")
        recs = get_recs(item['asin'])
        
        

        if not recs.empty:
            cols = st.columns(5)
            for i, (_, r) in enumerate(recs.iterrows()):
                with cols[i]:
                    with st.container(border=True):
                        st.image(r['hd_image'], use_container_width=True) 
                        st.caption(f"{r['clean_title'][:40]}...")
                        st.write(f":red[${r['price_numeric']:.2f}]")
                        # Nút Xem Ngay dùng Callback
                        st.button("Xem ngay", key=f"rec_{r['asin']}", 
                                 on_click=cb_view_product, args=(r['asin'],))
        else:
            st.warning("Không tìm thấy gợi ý (Lạ nhỉ, kiểm tra lại data).")

    except Exception as e:
        st.error(f"Lỗi hiển thị: {e}")
        st.button("Về trang chủ (Reset)", on_click=cb_go_home)

# --- TRANG CHỦ ---
else:
    st.subheader("🔥 Xu hướng & Gợi ý hôm nay")
    
    cols = st.columns(4)
    for i, (_, r) in enumerate(st.session_state.trends.iterrows()):
        with cols[i % 4]:
            with st.container(border=True):
                st.image(r['hd_image'], use_container_width=True)
                st.markdown(f"**{r['clean_title'][:50]}...**")
                st.write(f"💰 :red[${r['price_numeric']:.2f}]")
                
                # Nút xem dùng Callback
                st.button("Xem chi tiết", key=f"trend_{r['asin']}", 
                         use_container_width=True, 
                         on_click=cb_view_product, args=(r['asin'],))