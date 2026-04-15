# Hermes-MemoryDB

Hermes Agent 向量記憶資料庫 - 支援長期記憶、每日筆記、向量嵌入搜尋

## 資料庫架構

使用 SQLite + sqlite-vector 儲存向量嵌入

### 資料表

| 資料表 | 說明 |
|--------|------|
| **memories** | 長期記憶（對應 MEMORY.md） |
| **daily_notes** | 每日筆記（對應 memory/YYYY-MM-DD.md） |
| **memory_embeddings** | 向量嵌入（用於語意搜尋） |

### Schema

```sql
memories (
  id INTEGER PRIMARY KEY,
  category TEXT,      -- e.g., "project", "person", "preference"
  title TEXT,
  content TEXT,
  tags TEXT,          -- comma-separated tags
  created_at TIMESTAMP,
  updated_at TIMESTAMP
)

daily_notes (
  id INTEGER PRIMARY KEY,
  date TEXT UNIQUE,   -- YYYY-MM-DD
  content TEXT,
  created_at TIMESTAMP
)

memory_embeddings (
  id INTEGER PRIMARY KEY,
  memory_id INTEGER,  -- FK to memories.id
  memory_type TEXT,   -- 'memory' or 'daily_note'
  embedding BLOB,     -- 1536 dim vector (text-embedding-3-small)
  created_at TIMESTAMP
)
```

## 向量搜尋

使用 sqlite-vector 擴充元件進行相似度搜尋， fallback 到 LIKE 搜尋。

目前設定：
- Provider: MiniMax
- Model: text-embedding-3-small
- Dimension: 1536

## 管理指令

```bash
python3 memory_db.py init      # 初始化資料庫
python3 memory_db.py import     # 匯入 .md 檔案
python3 memory_db.py embed     # 生成所有記憶的向量嵌入
python3 memory_db.py search <query>  # 搜尋記憶
python3 memory_db.py config <provider> <api_url> <model> <dims>  # 設定 embedding provider
```
