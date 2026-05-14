import torch
from bertopic import BERTopic
from db import get_session, Comment
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def load_llm():
    print("正在載入本地 LLM (Qwen2.5-1.5B-Instruct)...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    device = "cpu" if torch.cuda.is_available() else "cuda"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        print(f"✅ 模型載入成功，使用設備: {device}")
        return tokenizer, model
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return None, None

def analyze_topics(video_id=None):
    session = get_session()
    
    # 撈取留言
    query = session.query(Comment)
    if video_id:
        query = query.filter_by(video_id=video_id)
        
    comments = query.all()
    
    if not comments or len(comments) < 10:
        print("沒有足夠的留言可以進行主題分析 (至少需要10筆)。")
        session.close()
        return

    from snownlp import SnowNLP

    print(f"準備對 {len(comments)} 筆留言進行 BERTopic 分群與情緒分析...")
    texts = [c.text for c in comments]

    print("正在執行情緒分析 (SnowNLP)...")
    for c in comments:
        if not c.text.strip():
            c.sentiment_score = 0.5
            c.sentiment_label = "中立"
            continue
        try:
            s = SnowNLP(c.text)
            score = s.sentiments
            c.sentiment_score = score
            if score >= 0.6:
                c.sentiment_label = "正向"
            elif score <= 0.4:
                c.sentiment_label = "負向"
            else:
                c.sentiment_label = "中立"
        except Exception:
            c.sentiment_score = 0.5
            c.sentiment_label = "中立"

    # 1. 初始化 BERTopic (自動處理 Embedding, UMAP, HDBSCAN, c-TF-IDF)
    # 語言設為 multilingual 支援中文
    topic_model = BERTopic(language="multilingual", calculate_probabilities=False, min_topic_size=5)
    
    # 執行分群
    topics, probs = topic_model.fit_transform(texts)
    
    print("分群完成，開始擷取關鍵字並請 LLM 命名主題...")
    
    # 取得 Topic 資訊
    topic_info = topic_model.get_topic_info()
    
    # 載入 LLM
    tokenizer, model = load_llm()
    if not model:
        session.close()
        return
        
    # 儲存每個 Topic 的名稱與關鍵字
    topic_mapping = {}
    
    for idx, row in topic_info.iterrows():
        topic_id = row['Topic']
        # -1 通常代表無法歸類的雜訊
        if topic_id == -1:
            topic_mapping[topic_id] = {
                "name": "其他/雜訊",
                "keywords": ""
            }
            continue
            
        # 取得該 Topic 的代表關鍵字
        keywords_list = [word for word, score in topic_model.get_topic(topic_id)]
        keywords_str = ", ".join(keywords_list)
        
        # 取得幾篇代表性留言
        representative_docs = topic_model.get_representative_docs(topic_id)
        docs_str = "\n".join([f"- {doc}" for doc in representative_docs[:3]])
        
        # Prompt 設計
        prompt = f"""你是一位專業的輿情分析師。我使用主題模型分出了一群相關的 YouTube 留言。
請根據以下「關鍵字」與「代表性留言」，為這個主題給出一個精簡的「主題名稱」（最多10個字，不要任何多餘的解釋或標點符號）。

【關鍵字】：{keywords_str}
【代表性留言】：
{docs_str}

【主題名稱】："""

        messages = [
            {"role": "system", "content": "你是一個有用的 AI 助手，專門負責分析繁體中文輿情。"},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=20,
            temperature=0.3,
            top_p=0.9
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        topic_name = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        topic_mapping[topic_id] = {
            "name": topic_name,
            "keywords": keywords_str
        }
        
        print(f"\n📊 Topic {topic_id}")
        print(f"🔹 關鍵字: {keywords_str}")
        print(f"🔹 命名結果: 【{topic_name}】")
        print(f"🔹 包含留言數: {row['Count']}")

    # 更新回資料庫
    print("\n正在將分析結果寫回資料庫...")
    for c, t_id in zip(comments, topics):
        c.topic_id = int(t_id)
        c.topic_name = topic_mapping[t_id]["name"]
        c.topic_keywords = topic_mapping[t_id]["keywords"]
        
    session.commit()
    session.close()
    print("主題模型分析與寫入完成！")
    
    print("\n" + "="*50)
    print("開始生成整體輿情總結大綱...")
    
    # 彙整所有主題資訊
    topics_summary_text = ""
    for t_id, info in topic_mapping.items():
        if t_id == -1:
            continue
        topics_summary_text += f"- 主題：{info['name']}\n  關鍵字：{info['keywords']}\n"
        
    if topics_summary_text:
        summary_prompt = f"""你是一位專業的輿情分析師。以下是這支 YouTube 影片留言經過主題模型分類後，萃取出的各個「子主題」與對應的「關鍵字」。
請根據這些資訊，寫出一份「整體輿情總結大綱」（約 3~5 點），幫助客戶快速掌握觀眾的整體反應與主要焦點。

【各子主題資訊】：
{topics_summary_text}

【整體輿情總結大綱】："""

        messages = [
            {"role": "system", "content": "你是一個有用的 AI 助手，專門負責分析繁體中文輿情。"},
            {"role": "user", "content": summary_prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        final_summary = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        print("\n📊 【影片留言整體輿情總結大綱】")
        print("-" * 30)
        print(final_summary)
        print("="*50)
    else:
        print("沒有足夠的明確主題可以生成總結。")

if __name__ == "__main__":
    analyze_topics()
