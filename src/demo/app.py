from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from src.demo.predict import predict_topk
from src.demo.load_models import load_models

st.set_page_config(
    page_title="Trí tuệ Nhân tạo sinh Văn bản", 
    page_icon="🤖", 
    layout="wide"
)

st.title("🤖 Cuộc đua AI: RNN vs LSTM")
st.markdown("Nhập một đoạn văn bản tiếng Việt và xem hai mô hình dự đoán từ tiếp theo như thế nào nhé!")

@st.cache_resource(show_spinner="Đang đánh thức AI... Lần đầu có thể mất 10 giây nhé! ☕")
def init_system():
    rnn_model, lstm_model, vocab, max_length = load_models()
    word2idx = vocab.get("word2idx", vocab) 
    
    if "idx2word" in vocab:
        idx2word = vocab["idx2word"]
    else:
        idx2word = {str(v): k for k, v in word2idx.items()}
        
    return rnn_model, lstm_model, word2idx, idx2word, max_length

rnn, lstm, word2idx, idx2word, max_length = init_system()

def show_topk(model_name, model, text, k=5):
    st.subheader(f"🧠 Mô hình {model_name}")
    
    try:
        results = predict_topk(model, text, word2idx, idx2word, max_length, k)
        
        # Vẽ các thanh xác suất (Progress Bar)
        for word, prob in results:
            col1, col2 = st.columns([1, 3]) # Cột 1 chứa chữ, cột 2 chứa thanh tỷ lệ
            with col1:
                st.markdown(f"**{word}**")
            with col2:
                # Ép prob về dải 0.0 - 1.0 cho thanh progress
                st.progress(prob, text=f"{prob*100:.2f}%")
                
    except Exception as e:
        st.error(f"Có lỗi xảy ra: {e}")

st.divider()

# Khung nhập liệu
user_text = st.text_input("✍️ Nhập câu mồi của bạn vào đây:", value="tôi đang học tập")

# Nút kích hoạt
if st.button("🚀 Dự đoán từ tiếp theo", use_container_width=True, type="primary"):
    if user_text.strip() == "":
        st.warning("Bạn chưa nhập chữ nào kìa!")
    else:
        # Chia màn hình làm 2 cột để so sánh trực quan
        col_rnn, col_lstm = st.columns(2)
        
        with col_rnn:
            show_topk('RNN', rnn, user_text, k=5)
            
        with col_lstm:
            show_topk('LSTM', lstm, user_text, k=5)

st.divider()
st.caption("Dự án xây dựng bằng TensorFlow/Keras và Streamlit.")