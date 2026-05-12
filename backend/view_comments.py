from db import get_session, Comment
import pandas as pd

def view_comments_by_topic():
    session = get_session()
    
    # 撈取所有留言 (包含分類為 -1 的雜訊)
    comments = session.query(Comment).all()
    
    if not comments:
        print("資料庫中沒有找到留言。請確認是否已執行爬蟲與分析。")
        session.close()
        return

    # 將資料轉為 DataFrame 方便分群與排序
    data = []
    for c in comments:
        data.append({
            "Topic ID": c.topic_id if c.topic_id is not None else -99,
            "主題名稱": c.topic_name if c.topic_name else "尚未分析",
            "關鍵字": c.topic_keywords if c.topic_keywords else "",
            "留言作者": c.author,
            "留言內容": c.text,
            "按讚數": c.like_count
        })
        
    df = pd.DataFrame(data)
    session.close()
    
    if df['Topic ID'].iloc[0] == -99:
        print("留言尚未經過主題分析，請先執行 python topic_modeling.py")
        return

    # 取得所有獨特的主題 ID 並排序
    topic_ids = sorted(df['Topic ID'].unique())
    
    print("\n" + "="*60)
    print(f"📊 留言分類檢視器：共找到 {len(topic_ids)} 個群組 (含雜訊)")
    print("="*60)
    
    for t_id in topic_ids:
        topic_df = df[df['Topic ID'] == t_id]
        topic_name = topic_df['主題名稱'].iloc[0]
        keywords = topic_df['關鍵字'].iloc[0]
        
        print(f"\n📁 【Topic {t_id}】: {topic_name} (共 {len(topic_df)} 則留言)")
        if t_id != -1:
            print(f"🔑 關鍵字: {keywords}")
        print("-" * 60)
        
        # 依照按讚數排序印出該主題的所有留言
        topic_df = topic_df.sort_values(by="按讚數", ascending=False)
        for idx, row in topic_df.iterrows():
            likes = f"(👍 {row['按讚數']})" if row['按讚數'] > 0 else ""
            print(f"👤 {row['留言作者']} {likes}")
            print(f"💬 {row['留言內容']}\n")

if __name__ == "__main__":
    # 提供一個簡單的互動介面
    print("1. 遍歷顯示所有主題與留言")
    print("2. 搜尋特定關鍵字的留言")
    choice = input("請選擇功能 (1 或 2): ")
    
    if choice == '1':
        view_comments_by_topic()
    elif choice == '2':
        keyword = input("請輸入搜尋關鍵字: ")
        session = get_session()
        # 利用 SQLite 基礎的 LIKE 搜尋
        results = session.query(Comment).filter(Comment.text.like(f"%{keyword}%")).all()
        print(f"\n🔍 找到 {len(results)} 筆包含「{keyword}」的留言：\n")
        for r in results:
            print(f"[{r.topic_name}] 👤 {r.author}: {r.text}")
        session.close()
    else:
        print("無效的選擇。")
