import torch
from transformers import pipeline
from db import get_session, Comment
from tqdm import tqdm

def load_sentiment_model():
    print("正在載入情緒分析模型...")
    # 檢查是否有 GPU 可用
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print(f"✅ 偵測到 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 未偵測到 GPU，將使用 CPU 進行推論。")
        
    # 使用 IDEA-CCNL 開源的中文情緒分析模型 (基於 Roberta)
    # 它對於中文社群平台的文字有不錯的判斷能力
    model_name = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
    try:
        classifier = pipeline("sentiment-analysis", model=model_name, device=device)
        return classifier
    except Exception as e:
        print(f"模型載入失敗: {e}")
        return None

def analyze_sentiments(batch_size=32):
    session = get_session()
    
    # 撈取尚未有情緒分數的留言
    unprocessed_comments = session.query(Comment).filter(Comment.sentiment_label == None).all()
    
    if not unprocessed_comments:
        print("目前沒有需要處理的留言。")
        session.close()
        return

    print(f"找到 {len(unprocessed_comments)} 筆未處理留言，準備進行情緒分析...")
    
    classifier = load_sentiment_model()
    if not classifier:
        session.close()
        return

    # 將留言轉為純文字陣列
    texts = [c.text for c in unprocessed_comments]
    
    # 由於留言可能過長，做一個簡單的長度截斷保護
    max_length = 512
    texts = [t[:max_length] for t in texts]

    results = []
    print("開始批次推論...")
    # 使用 pipeline 進行批次預測
    for out in tqdm(classifier(texts, batch_size=batch_size, truncation=True)):
        results.append(out)

    # 寫回資料庫
    print("將結果寫回資料庫...")
    for comment, result in zip(unprocessed_comments, results):
        # 標籤通常是 'Positive', 'Negative' 等
        comment.sentiment_label = result['label']
        comment.sentiment_score = float(result['score'])
    
    session.commit()
    session.close()
    print("情緒分析處理完成！")

if __name__ == "__main__":
    analyze_sentiments()
