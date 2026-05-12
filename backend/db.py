import os
from sqlalchemy import create_engine, Column, String, Text, Integer, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# 設定 SQLite 資料庫檔案路徑，預設存在 backend 資料夾內
DB_PATH = os.path.join(os.path.dirname(__file__), 'comments.db')
engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)

Base = declarative_base()

class Comment(Base):
    __tablename__ = 'comments'

    comment_id = Column(String, primary_key=True, comment="YouTube 留言 ID")
    video_id = Column(String, index=True, nullable=False, comment="YouTube 影片 ID")
    author = Column(String, comment="留言作者名稱")
    text = Column(Text, nullable=False, comment="清理後的留言內容")
    like_count = Column(Integer, default=0, comment="按讚數")
    published_at = Column(DateTime, comment="發布時間")
    
    # 後續 NLP (Topic Modeling) 處理的欄位
    topic_id = Column(Integer, nullable=True, index=True, comment="BERTopic 分群 ID (-1 表雜訊)")
    topic_keywords = Column(String, nullable=True, comment="該群組的代表性關鍵字")
    topic_name = Column(String, nullable=True, comment="由 LLM 命名的主題名稱")
    
    # 紀錄抓取時間
    created_at = Column(DateTime, default=datetime.utcnow, comment="寫入資料庫時間")

# 建立所有的資料表
Base.metadata.create_all(engine)

# 建立 Session 工廠
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_session():
    return SessionLocal()
