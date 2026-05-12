# YouTube 留言輿情主題分析系統 (YouTube Comments Topic Modeling Analyzer)

這是一個針對 YouTube 影片留言的自動化輿情分析系統。跳脫傳統單純的「正負向情緒分析」，本專案透過 **BERTopic 主題模型** 與 **本地端大型語言模型 (Qwen2.5-1.5B-Instruct)** 的完美結合，能自動將海量留言進行語意分群、萃取關鍵字，並由 LLM 自動為每個群組命名，最終產出一份精簡的「整體輿情總結大綱」。

全系統採 **100% 本地端執行** 設計，無需依賴昂貴的商業 API (如 OpenAI)，確保資料隱私且零後續推論成本。

## 🌟 核心功能

1. **自動化資料採集**：透過 YouTube Data API v3 精準抓取熱門留言，並自動清洗髒資料 (如網址、多餘空白)。
2. **高效能主題分群 (BERTopic)**：
   - 採用 `paraphrase-multilingual-MiniLM-L12-v2` 進行跨語言/中文語意向量化。
   - 底層透過 HDBSCAN + UMAP，物理性地將相似留言聚類，並自動過濾無意義的雜訊酸言酸語。
3. **自動命名與總結 (Local LLM)**：
   - 搭載阿里開源的 `Qwen2.5-1.5B-Instruct` 輕量模型（僅需 ~4GB VRAM）。
   - 根據群組的 c-TF-IDF 關鍵字，自動為每個主題命名。
   - 總覽所有主題，生成 3~5 點的「整體輿情總結大綱」。
4. **互動式檢視工具**：內建命令列檢視器，支援遍歷特定主題下的熱門留言，並提供關鍵字檢索功能。

---

## 🏗️ 系統架構

專案採模組化設計，包含以下核心組件：

*   `backend/scraper.py`：負責串接 API 爬取留言，並將資料結構化寫入 SQLite 資料庫 (`comments.db`)。
*   `backend/topic_modeling.py`：核心分析引擎。執行 Embedding -> Clustering -> Keyword Extraction -> LLM 命名，並把結果更新回資料庫。
*   `backend/view_comments.py`：檢視器。提供終端機介面讓使用者輕鬆瀏覽分群後的留言與關鍵字。
*   `generate_demo.py`：報表產生器。可自動輸出一份完整的 `.ipynb` Demo 檔案。

---

## 💻 環境建置與執行指南

本專案強烈建議在具備 **NVIDIA GPU (CUDA)** 的環境下執行，以獲得極致的分析速度（千筆留言約 1 分鐘內處理完畢）。

### 1. 安裝套件
請確認您的環境中已安裝 Python 3.9+，並依照您的 CUDA 版本安裝對應的 PyTorch：
```bash
# 1. 建立虛擬環境 (建議)
python -m venv backend/venv
source backend/venv/Scripts/activate  # Windows PowerShell

# 2. 安裝相依套件 (包含 BERTopic, Transformers, Pandas 等)
pip install -r backend/requirements.txt
```

### 2. 設定環境變數
請在 `backend` 資料夾中複製 `.env.example` 並重新命名為 `.env`，填入您的 API Key：
```env
YOUTUBE_API_KEY="您的_Google_Cloud_API_Key"
```

### 3. 執行分析流程
請依序執行以下指令：
```bash
# Step 1: 抓取留言
python backend/scraper.py

# Step 2: 執行主題模型與 LLM 總結 (初次執行會自動下載 HuggingFace 模型權重)
python backend/topic_modeling.py

# Step 3: 開啟互動式留言檢視器
python backend/view_comments.py
```

---

## 📈 為什麼選擇這個架構？

*   **為何不使用傳統 LDA？** LDA 基於詞頻，無法理解社群媒體中充滿口語化、短文本的留言語意。BERTopic 基於 Transformer 語意向量，能精準捕捉上下文意圖。
*   **為何選擇 Qwen 1.5B？** 一般 8B 等級以上的模型需要至少 8GB~16GB 的 VRAM，不利於一般筆電部署。Qwen 1.5B 在極低的硬體門檻下，展現了驚人的繁體中文理解力與 Zero-shot 摘要能力。

## 授權條款
MIT License
