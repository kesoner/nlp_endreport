import os
import re
import json
import urllib.parse
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime
from db import get_session, Comment

# 載入環境變數
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def extract_video_id(url: str) -> str:
    """從 YouTube 網址中萃取 Video ID"""
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return urllib.parse.parse_qs(parsed_url.query).get('v', [None])[0]
        if parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        if parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]
    return None

def clean_comment(text: str) -> str:
    """清理留言內容：移除 URL、多餘空白、純換行"""
    # 移除 URL
    text = re.sub(r'http[s]?://\S+', '', text)
    # 移除多餘空白與換行
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_top_comments(video_url: str, max_results: int = 1000) -> list:
    """抓取 YouTube 影片的熱門留言"""
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("無效的 YouTube 網址")
        
    if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == "your_api_key_here":
        raise ValueError("請在 .env 檔案中設定 YOUTUBE_API_KEY")

    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    
    comments = []
    next_page_token = None
    
    print(f"開始抓取影片 {video_id} 的留言...")
    
    try:
        while len(comments) < max_results:
            # 每次請求最多 100 筆
            request_count = min(100, max_results - len(comments))
            
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=request_count,
                order="relevance", # 依熱門程度排序
                textFormat="plainText",
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                raw_text = snippet['textDisplay']
                cleaned_text = clean_comment(raw_text)
                
                # 過濾掉清理後為空的留言（例如原本只有網址的留言）
                if not cleaned_text:
                    continue
                    
                comments.append({
                    'comment_id': item['id'],
                    'video_id': video_id,
                    'author': snippet['authorDisplayName'],
                    'text': cleaned_text,
                    'like_count': snippet['likeCount'],
                    'published_at': snippet['publishedAt']
                })
                
                if len(comments) >= max_results:
                    break
                    
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break # 沒有下一頁了
                
        print(f"成功抓取 {len(comments)} 筆留言。")
        return comments
        
    except HttpError as e:
        print(f"發生 API 錯誤: {e}")
        return []

if __name__ == "__main__":
    # 測試用網址 (可以替換成你想測試的影片)
    test_url = "https://www.youtube.com/watch?v=UIVtl1AclxA"
    
    try:
        # 將 max_results 設為一個極大的數字(例如 999999)，就可以不斷翻頁直到抓完所有留言。
        # 注意：若影片留言高達數萬筆，會花費較長的時間，且會消耗較多 YouTube API 的每日額度。
        results = fetch_top_comments(test_url, max_results=999999)
        
        if results:
            session = get_session()
            
            # 清空舊的留言紀錄，確保每次分析都是針對單一影片
            print("正在清空舊資料庫...")
            try:
                deleted_count = session.query(Comment).delete()
                session.commit()
                print(f"已清空 {deleted_count} 筆歷史留言。")
            except Exception as e:
                session.rollback()
                print(f"清空資料庫時發生錯誤: {e}")

            new_count = 0
            for item in results:
                # 檢查資料庫是否已存在該留言
                existing = session.query(Comment).filter_by(comment_id=item['comment_id']).first()
                if not existing:
                    # 解析 YouTube API 回傳的 ISO 8601 時間字串
                    # 例如 "2009-10-25T06:57:33Z"
                    pub_time = datetime.strptime(item['published_at'], "%Y-%m-%dT%H:%M:%SZ")
                    
                    comment_db = Comment(
                        comment_id=item['comment_id'],
                        video_id=item['video_id'],
                        author=item['author'],
                        text=item['text'],
                        like_count=item['like_count'],
                        published_at=pub_time
                    )
                    session.add(comment_db)
                    new_count += 1
            
            session.commit()
            session.close()
            print(f"成功將 {new_count} 筆新留言寫入資料庫。")
        else:
            print("沒有抓取到任何留言。")
        
    except Exception as e:
        print(f"執行失敗: {e}")
