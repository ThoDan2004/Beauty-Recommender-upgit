import streamlit as st
import pandas as pd
import numpy as np
import re
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

# --- 1. CẤU HÌNH & STATE ---
st.set_page_config(layout="wide", page_title="Women's Beauty & Jewelry", page_icon="💎")

if 'history' not in st.session_state: st.session_state.history = []
if 'favorites' not in st.session_state: st.session_state.favorites = []
if 'view_asin' not in st.session_state: st.session_state.view_asin = None 
if 'trends' not in st.session_state: st.session_state.trends = None
if 'search_results' not in st.session_state: st.session_state.search_results = None 

# --- 2. TỪ ĐIỂN THÔNG MINH (VIETNAMESE MAPPING) ---
VIET_TO_ENG = {
    'son': 'Lipstick', 'môi': 'Lip', 'nhẫn': 'Ring', 
    'nền': 'Foundation', 'phấn': 'Powder', 'má hồng': 'Blush',
    'mắt': 'Eye', 'mi': 'Lash', 'mascara': 'Mascara', 'kẻ mắt': 'Eyeliner',
    'mày': 'Brow', 'che khuyết điểm': 'Concealer', 'trang điểm': 'Makeup',
    'cọ': 'Brush', 'mút': 'Sponge', 'tẩy trang': 'Remover Cleanser',
    'kem': 'Cream', 'dưỡng': 'Lotion Moisturizer', 'serum': 'Serum',
    'mặt nạ': 'Mask', 'rửa mặt': 'Cleanser Wash', 'nước hoa hồng': 'Toner',
    'tẩy da chết': 'Scrub Exfoliator', 'chống nắng': 'Sunscreen Sunblock',
    'mụn': 'Acne', 'lão hóa': 'Anti-aging', 'nhăn': 'Wrinkle',
    'trắng': 'Whitening Brightening', 'nám': 'Spot', 'thâm': 'Dark',
    'cấp ẩm': 'Hydrating', 'dầu': 'Oil',
    'nước hoa': 'Perfume Fragrance', 'dầu thơm': 'Fragrance',
    'gội': 'Shampoo', 'xả': 'Conditioner', 'tóc': 'Hair',
    'nhuộm': 'Color', 'sấy': 'Dryer', 'duỗi': 'Straightener', 'uốn': 'Curler',
    'tắm': 'Bath', 'xà phòng': 'Soap', 'sữa tắm': 'Wash', 'body': 'Body',
    'lông': 'Hair Removal', 'cạo': 'Shaver Razor', 'khử mùi': 'Deodorant',
    'nâu da': 'Tanning', 'móng': 'Nail', 'sơn móng': 'Polish',
    'dây chuyền': 'Necklace', 'vòng cổ': 'Necklace',
    'bông tai': 'Earring', 'khuyên': 'Earring', 'hoa tai': 'Earring',
    'lắc': 'Bracelet', 'vòng tay': 'Bracelet', 'vòng': 'Bracelet',
    'mặt dây': 'Pendant', 'lắc chân': 'Anklet', 'trâm': 'Hairpin',
    'bạc': 'Silver', 'vàng': 'Gold', 'kim cương': 'Diamond', 
    'ngọc trai': 'Pearl', 'đá': 'Gemstone', 'pha lê': 'Crystal',
    'lọ': 'Container Jar', 'hũ': 'Jar', 'chai': 'Bottle',
    'gương': 'Mirror', 'kéo': 'Scissor', 'nhíp': 'Tweezer',
    'bông': 'Cotton', 'khăn': 'Towel Wipes', 'máy': 'Machine Electric'
}

def smart_translate(query):
    """Dịch từ khóa Việt -> Anh và loại bỏ từ rác"""
    if not query: return ""
    query_lower = query.lower()
    
    # Tìm các từ khóa có trong từ điển
    found_keywords = []
    # Sắp xếp từ dài đến ngắn để ưu tiên từ ghép (ví dụ 'nước hoa' ưu tiên hơn 'hoa')
    sorted_keys = sorted(VIET_TO_ENG.keys(), key=len, reverse=True)
    
    temp_query = query_lower
    for vn_word in sorted_keys:
        if vn_word in temp_query:
            found_keywords.append(VIET_TO_ENG[vn_word])
            # Xóa từ đã tìm thấy khỏi chuỗi để tránh lặp
            temp_query = temp_query.replace(vn_word, " ")
            
    # Nếu tìm thấy từ khóa, trả về danh sách từ khóa tiếng Anh
    if found_keywords:
        return " ".join(found_keywords)
    
    # Nếu không tìm thấy gì (ví dụ tên riêng tiếng Anh: 'Olay'), trả về nguyên gốc
    return query

# --- 3. HÀM DỊCH MÔ TẢ & XỬ LÝ TEXT ---
@st.cache_data(show_spinner=False)
def translate_description(text):
    try:
        if len(text) > 4500: text = text[:4500]
        translator = GoogleTranslator(source='auto', target='vi')
        return translator.translate(text)
    except:
        return "Hệ thống dịch đang bận, vui lòng thử lại sau."

def clean_text_display(text):
    if pd.isna(text): return "Thông tin đang cập nhật..."
    text = str(text)
    if text.startswith("['") and text.endswith("']"): text = text[2:-2]
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    return " ".join(text.split())

def get_hd_image(url):
    if pd.isna(url) or 'http' not in str(url): return "https://via.placeholder.com/300x400?text=No+Image"
    hd_url = re.sub(r'\._[A-Z]{2}\d+(,_\d+)?_(\.[a-z]+)$', r'\2', str(url))
    hd_url = re.sub(r'\._AC_.*(\.[a-z]+)$', r'\1', hd_url)
    return hd_url

# --- 4. LOAD DỮ LIỆU ---
@st.cache_resource
def load_data_and_model():
    input_file = 'Women_Cosmetics_Jewelry_Clean.csv'
    try:
        df = pd.read_csv(input_file)
    except:
        st.error(f"Chưa có file '{input_file}'. Hãy chạy Preprocessing trước!")
        st.stop()
        
    df = df.drop_duplicates(subset=['asin']).copy()
    df = df.reset_index(drop=True)
    
    df['clean_desc'] = df['item_text'].apply(clean_text_display)
    df['clean_title'] = df['title'].apply(lambda x: html.unescape(str(x)))
    df['hd_image'] = df['image_url_clean'].apply(get_hd_image)
    if 'price_numeric' not in df.columns: df['price_numeric'] = 0.0
    
    df['training_text'] = (df['clean_title'] + " " + df['clean_title'] + " " + df['clean_title'] + " " + df['item_text']).fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    tfidf_matrix = tfidf.fit_transform(df['training_text'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, cosine_sim

df, cosine_sim = load_data_and_model()

if st.session_state.trends is None:
    st.session_state.trends = df.sample(min(12, len(df)))

# --- 5. LOGIC CALLBACK (CORE FIXES) ---
def cb_view_product(asin):
    st.session_state.view_asin = asin
    st.session_state.search_results = None 
    item = df[df['asin'] == asin].iloc[0]
    st.session_state.history = [h for h in st.session_state.history if h['asin'] != asin]
    st.session_state.history.insert(0, {'asin': asin, 'title': item['clean_title']})
    st.session_state.history = st.session_state.history[:15]

def cb_go_home():
    st.session_state.view_asin = None
    st.session_state.search_results = None
    st.session_state.search_query = "" 

def cb_toggle_favorite(asin):
    if asin in st.session_state.favorites:
        st.session_state.favorites.remove(asin)
    else:
        st.session_state.favorites.append(asin)

# --- QUAN TRỌNG: LOGIC TÌM KIẾM MỀM DẺO ---
def cb_search():
    query = st.session_state.search_query
    if query:
        # 1. Dịch từ khóa
        translated_query = smart_translate(query)
        
        # 2. Tách từ khóa thành các từ đơn (Ví dụ "Cream Eye" -> ["Cream", "Eye"])
        keywords = translated_query.split()
        
        # 3. Lọc: Sản phẩm phải chứa TẤT CẢ các từ khóa (Logic AND)
        # Giúp tìm "Kem mắt" -> phải có cả "Cream" và "Eye"
        mask = np.ones(len(df), dtype=bool)
        for kw in keywords:
            mask = mask & df['clean_title'].str.contains(kw, case=False, na=False)
        
        results = df[mask]
        
        # 4. Nếu tìm kỹ không thấy, thử tìm lỏng lẻo (Logic OR)
        # Tìm sản phẩm chứa ÍT NHẤT 1 từ khóa
        if results.empty and len(keywords) > 1:
            mask_or = np.zeros(len(df), dtype=bool)
            for kw in keywords:
                mask_or = mask_or | df['clean_title'].str.contains(kw, case=False, na=False)
            results = df[mask_or]
            
            # Nếu tìm thấy theo cách lỏng lẻo, thông báo nhẹ
            if not results.empty:
                st.toast(f"Không tìm thấy chính xác '{translated_query}', hiển thị kết quả gần đúng.")

        # Nếu vẫn không thấy, thử tìm trong Brand
        if results.empty:
             results = df[df['brand'].str.contains(translated_query, case=False, na=False)]

        if not results.empty:
            st.session_state.search_results = results
        else:
            st.toast(f"Không tìm thấy: '{query}' (Dịch: {translated_query})")
            st.session_state.search_results = pd.DataFrame()

def get_recs(asin, top_k=5):
    try:
        idx = df.index[df['asin'] == asin].tolist()[0]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_k+1]
        item_indices = [i[0] for i in sim_scores]
        return df.iloc[item_indices]
    except:
        return pd.DataFrame()

# --- 6. GIAO DIỆN ---
with st.sidebar:
    st.title("🎀 Menu")
    st.button("🏠 Trang chủ", use_container_width=True, type="primary", on_click=cb_go_home)
    
    tab1, tab2 = st.tabs(["❤️ Yêu thích", "🕒 Lịch sử"])
    with tab1:
        if st.session_state.favorites:
            fav_items = df[df['asin'].isin(st.session_state.favorites)]
            for _, item in fav_items.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([1, 2])
                    c1.image(item['hd_image'])
                    c2.caption(item['clean_title'][:40])
                    c2.button("Xem", key=f"fav_v_{item['asin']}", on_click=cb_view_product, args=(item['asin'],))
                    c2.button("Xóa", key=f"fav_d_{item['asin']}", on_click=cb_toggle_favorite, args=(item['asin'],))
        else: st.info("Trống")
        
    with tab2:
        if st.session_state.history:
            for h in st.session_state.history:
                st.button(f"👁️ {h['title'][:25]}...", key=f"hist_{h['asin']}", use_container_width=True, on_click=cb_view_product, args=(h['asin'],))
            st.button("🗑️ Xóa lịch sử", on_click=lambda: st.session_state.update(history=[]))

# --- 7. MAIN ---
st.title("💎 Women's Cosmetics & Jewelry Store")

c_search, c_btn = st.columns([4, 1])
with c_search:
    st.text_input(
        "🔍 Tìm kiếm (Tiếng Việt/Anh):", 
        key="search_query", 
        on_change=cb_search,
        placeholder="Gõ 'kem mắt', 'dưỡng môi', 'nhẫn vàng'..."
    )
with c_btn:
    st.write("") 
    st.write("") 
    st.button("Tìm", on_click=cb_search, type="primary")

st.divider()

if st.session_state.view_asin:
    try:
        item = df[df['asin'] == st.session_state.view_asin].iloc[0]
        st.button("⬅️ Quay lại", on_click=cb_go_home)

        c1, c2 = st.columns([1, 1.5])
        with c1: st.image(item['hd_image'], width=500)
        with c2:
            st.header(item['clean_title'])
            st.markdown(f"🏷️ **Thương hiệu:** {item['brand']}")
            st.subheader(f"💵 Giá: :red[${item['price_numeric']:.2f}]")
            
            is_fav = item['asin'] in st.session_state.favorites
            btn_lbl = "❤️ Bỏ thích" if is_fav else "🤍 Yêu thích"
            btn_typ = "primary" if is_fav else "secondary"
            st.button(btn_lbl, type=btn_typ, on_click=cb_toggle_favorite, args=(item['asin'],))

            with st.container(border=True):
                st.markdown("**📝 Mô tả sản phẩm:**")
                tab_en, tab_vn = st.tabs(["🇬🇧 English", "🇻🇳 Tiếng Việt (AI Dịch)"])
                with tab_en: st.write(item.get('clean_desc', item['item_text']))
                with tab_vn:
                    with st.spinner("Đang dịch..."):
                        raw_text = item.get('clean_desc', item['item_text'])
                        st.write(translate_description(raw_text))
        
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
                        st.button("Xem ngay", key=f"rec_{r['asin']}", on_click=cb_view_product, args=(r['asin'],))
        else: st.warning("Không tìm thấy gợi ý.")
    except Exception as e:
        st.error("Lỗi hiển thị."); st.button("Reset", on_click=cb_go_home)

elif st.session_state.search_results is not None:
    results = st.session_state.search_results
    st.subheader(f"🔎 Kết quả tìm kiếm ({len(results)} sản phẩm)")
    if not results.empty:
        display_items = results.head(20)
        cols = st.columns(4)
        for i, (_, r) in enumerate(display_items.iterrows()):
            with cols[i % 4]:
                with st.container(border=True):
                    st.image(r['hd_image'], use_container_width=True)
                    st.markdown(f"**{r['clean_title'][:50]}...**")
                    st.write(f"💵 :red[${r['price_numeric']:.2f}]")
                    st.button("Xem chi tiết", key=f"search_{r['asin']}", on_click=cb_view_product, args=(r['asin'],), use_container_width=True)
    else: st.info("Không tìm thấy sản phẩm phù hợp.")

else:
    st.subheader("🔥 Xu hướng & Gợi ý hôm nay")
    cols = st.columns(4)
    for i, (_, r) in enumerate(st.session_state.trends.iterrows()):
        with cols[i % 4]:
            with st.container(border=True):
                st.image(r['hd_image'], use_container_width=True)
                st.markdown(f"**{r['clean_title'][:50]}...**")
                st.write(f"💰 :red[${r['price_numeric']:.2f}]")
                st.button("Xem chi tiết", key=f"trend_{r['asin']}", use_container_width=True, on_click=cb_view_product, args=(r['asin'],))