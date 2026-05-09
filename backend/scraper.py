import os
import re
import json
import urllib.parse
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    try:
        # 為了測試方便，這裡先抓 50 筆
        results = fetch_top_comments(test_url, max_results=50)
        
        # 存檔檢查結果
        output_file = "sample_comments.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"結果已儲存至 {output_file}")
        
    except Exception as e:
        print(f"執行失敗: {e}")
