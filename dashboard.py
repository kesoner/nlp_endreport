import streamlit as st
import pandas as pd
import sqlite3
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import textwrap

# 設定頁面標題
st.set_page_config(page_title="YouTube 留言輿情分析儀表板", layout="wide")

st.title("📊 YouTube 留言輿情分析儀表板")
st.markdown("本儀表板整合了 BERTopic 主題分群、本地 LLM 摘要以及 SnowNLP 情緒分析，幫助您快速掌握觀眾的核心反饋。")

# 讀取資料庫
@st.cache_data(ttl=10) # 設定快取時間 10 秒，方便更新
def load_data():
    db_path = os.path.join(os.path.dirname(__file__), 'backend', 'comments.db')
    if not os.path.exists(db_path):
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM comments", conn)
    conn.close()
    return df

df = load_data()

if df.empty:
    st.warning("⚠️ 資料庫中尚無資料，請先在終端機執行 `python backend/scraper.py` 與 `python backend/topic_modeling.py` 抓取與分析資料！")
    st.stop()

# ==================== 1. 總覽數據 ====================
st.header("1. 留言數據總覽")
col1, col2, col3 = st.columns(3)
col1.metric("總留言數", len(df))

analyzed_df = df[df['topic_id'] != -1].copy() if 'topic_id' in df.columns else pd.DataFrame()
noise_count = len(df[df['topic_id'] == -1]) if 'topic_id' in df.columns else 0
col2.metric("有效分析留言數", len(analyzed_df))
col3.metric("過濾雜訊數", noise_count)

st.divider()

# ==================== 2. 情緒圓餅圖 ====================
st.header("2. 整體情緒分佈 (Sentiment Analysis)")

if 'sentiment_label' in df.columns and not df['sentiment_label'].isnull().all():
    sentiment_counts = df['sentiment_label'].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {'正向': '#28a745', '中立': '#ffc107', '負向': '#dc3545'}
    
    # 確保字體支援中文 (Windows 預設微軟正黑體)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    ax.pie(sentiment_counts, 
           labels=sentiment_counts.index, 
           autopct='%1.1f%%', 
           startangle=90, 
           colors=[colors.get(x, '#999999') for x in sentiment_counts.index],
           textprops={'fontsize': 14})
    ax.axis('equal')  
    
    col_pie, _ = st.columns([1, 1])
    with col_pie:
        st.pyplot(fig)
else:
    st.info("尚無情緒分析資料。請確認是否已重新執行 `topic_modeling.py`。")

st.divider()

# ==================== 3. 關鍵字雲 ====================
st.header("3. 核心關鍵字雲 (Keyword Cloud)")

if not analyzed_df.empty and 'topic_keywords' in analyzed_df.columns:
    # 將所有 topic_keywords 串接在一起
    all_keywords = " ".join(analyzed_df['topic_keywords'].dropna().astype(str).tolist())
    # 把逗號取代為空白，方便文字雲切割
    all_keywords = all_keywords.replace(',', ' ')
    
    if all_keywords.strip():
        # Windows 預設中文字體路徑，若無則留空讓它找預設
        font_path = "C:/Windows/Fonts/msjh.ttc" if os.path.exists("C:/Windows/Fonts/msjh.ttc") else None
        
        try:
            wordcloud = WordCloud(
                font_path=font_path,
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(all_keywords)
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.imshow(wordcloud, interpolation='bilinear')
            ax2.axis('off')
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"產生文字雲時發生錯誤 (可能是系統缺少中文字體): {e}")
    else:
        st.info("沒有擷取到有效的關鍵字。")
else:
    st.info("尚無主題分群資料可以生成文字雲。")

st.divider()

# ==================== 4. 論點摘要卡片 ====================
st.header("4. 論點摘要卡片 (Topic Summaries)")

if not analyzed_df.empty and 'topic_name' in analyzed_df.columns:
    # 依照主題分組
    topics = analyzed_df.groupby(['topic_id', 'topic_name', 'topic_keywords']).size().reset_index(name='count')
    topics = topics.sort_values(by='count', ascending=False)
    
    for idx, row in topics.iterrows():
        t_id = row['topic_id']
        t_name = row['topic_name']
        t_keys = row['topic_keywords']
        t_count = row['count']
        
        with st.expander(f"📌 【{t_name}】 (共討論 {t_count} 次)", expanded=True):
            st.markdown(f"**🔑 核心關鍵字：** `{t_keys}`")
            
            # 取出該主題按讚數前 3 高的留言作為代表性論述
            top_comments = analyzed_df[analyzed_df['topic_id'] == t_id].sort_values(by='like_count', ascending=False).head(3)
            
            st.markdown("💬 **高讚數代表性留言：**")
            for _, c_row in top_comments.iterrows():
                likes = f"👍 {c_row['like_count']}" if c_row['like_count'] > 0 else "無按讚"
                sentiment = c_row.get('sentiment_label', '未分析')
                
                # 決定情緒標籤的顏色
                color = "green" if sentiment == "正向" else "red" if sentiment == "負向" else "orange"
                
                st.markdown(f"""
                <div style="padding:10px; border-radius:5px; background-color:#262730; margin-bottom:10px;">
                    <span style="font-size:0.9em; color:#ddd;">👤 <b>{c_row['author']}</b> ({likes})</span>
                    <span style="font-size:0.8em; color:{color}; padding:2px 6px; border:1px solid {color}; border-radius:10px; margin-left:10px;">{sentiment}</span>
                    <br><br>
                    <span style="font-size:1.1em; color:#fff;">{c_row['text']}</span>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("尚無主題論述資料。")
