# Hermes-MemoryDB

Hermes Agent 向量記憶資料庫 - PostgreSQL PG15 + pgvector

## 資料庫連線

連線資訊儲存於環境變數，請參閱內部文件。

## Schema: Rex-625081

### daily_log
每日筆記（對應 memory/YYYY-MM-DD.md）

| 欄位 | 類型 | 說明 |
|------|------|------|
| id | INTEGER | 主鍵 |
| log_date | DATE | 日期 (YYYY-MM-DD) |
| content | TEXT | 原始 markdown 內容 |
| metadata | JSONB | 擴充metadata |
| created_at | TIMESTAMP | 建立時間 |

### longterm_memory
長期記憶（對應 MEMORY.md）

| 欄位 | 類型 | 說明 |
|------|------|------|
| id | INTEGER | 主鍵 |
| content | TEXT | 記憶內容 |
| embedding | VECTOR | 向量嵌入 (pgvector, 1536 dim) |
| metadata | JSONB | 擴充metadata |
| created_at | TIMESTAMP | 建立時間 |

## 向量搜尋

使用 pgvector 擴充進行相似度搜尋。

目前設定：
- Provider: MiniMax
- Model: text-embedding-3-small
- Dimension: 1536
