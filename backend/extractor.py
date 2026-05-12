import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from db import get_session, Comment

def load_llm():
    print("正在載入本地 LLM (Qwen2.5-1.5B-Instruct)...")
    # 選擇輕量級且支援中文的開源模型，適合消費級 GPU (約需 3~4GB VRAM)
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 載入 Tokenizer 與模型
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

def extract_arguments(video_id=None, top_k=50):
    session = get_session()
    
    # 撈取留言
    query = session.query(Comment)
    if video_id:
        query = query.filter_by(video_id=video_id)
        
    # 先撈出所有留言，按讚數排序
    comments = query.order_by(Comment.like_count.desc()).all()
    session.close()
    
    if not comments:
        print("沒有找到可以分析的留言。")
        return
        
    # 根據情緒分類分組
    sentiment_groups = {}
    for c in comments:
        label = c.sentiment_label or "未分類"
        # 將英文標籤轉為中文顯示比較友善，例如 model 若回傳 'Positive' / 'Negative'
        if label.lower() == 'positive':
            display_label = "正向"
        elif label.lower() == 'negative':
            display_label = "負向"
        else:
            display_label = label
            
        if display_label not in sentiment_groups:
            sentiment_groups[display_label] = []
        sentiment_groups[display_label].append(c)
        
    tokenizer, model = load_llm()
    if not model:
        return

    print(f"\n準備分析影片 {video_id or '所有'} 的留言，共分為 {len(sentiment_groups)} 種情緒類別...")
    print("="*50)
    
    for label, group_comments in sentiment_groups.items():
        # 每個分類取前 top_k 筆高關注度留言來分析
        top_comments = group_comments[:top_k]
        print(f"正在分析情緒類別：【{label}】 (擷取該類別前 {len(top_comments)} 筆高關注留言)")
        
        # 將留言組合成 Prompt 內容
        comments_text = "\n".join([f"- {c.text}" for c in top_comments])
        
        prompt = f"""你是一位專業的輿情分析師。請閱讀以下來自 YouTube 影片的留言，這些留言已經被分類為「{label}」情緒。
請歸納出這些具備「{label}」情緒的觀眾的「主要論點」與「關注焦點」。
請用條列式（3~5點）輸出，並保持客觀中立。

【留言列表】：
{comments_text}

【分析報告】："""

        messages = [
            {"role": "system", "content": "你是一個有用的 AI 助手，專門負責分析繁體中文輿情。"},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
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
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"\n📊 【{label}】留言論點大綱")
        print("-" * 30)
        print(response)
        print("="*50)

if __name__ == "__main__":
    # 測試執行，您可以傳入特定的 video_id
    extract_arguments("aI6ChX9kbvs")
