# RAG Code QA — 智能知识库问答系统

基于 LangChain + DeepSeek 的检索增强生成（RAG）知识库问答系统，支持多格式技术文档上传与自然语言问答。

## 核心特性

- **混合检索**：Dense（FAISS）+ Sparse（BM25 + jieba 中文分词）双路召回 → RRF 融合 → BGE-Reranker 精排，四阶段检索流水线
- **查询优化**：LLM 驱动的 Query Rewriting + 复杂问题拆解，支持多轮对话消解指代
- **双模型验证**：Fast 模型生成答案 → Pro 模型评估 Faithfulness / Completeness / Relevance 并检测幻觉 → 低分自动重生成
- **多策略分块**：固定大小 / Markdown 层级感知 / Embedding 语义相似度 / AST 代码感知，四种策略按文档类型适配
- **元数据过滤**：检索前按文件类型筛选，检索后按查询意图加权重排
- **多后端支持**：FAISS / ChromaDB 双向量数据库可切换
- **Prompt 工程**：YAML 模板化管理，含 Few-shot + CoT，支持版本控制与 A/B 测试
- **自动化评测**：集成 RAGAS 评测框架

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 3. 预下载模型（可选，解决网络问题）
python download_models.py --local ./models
# 在 .env 中设置 LOCAL_MODEL_DIR=./models

# 4. 启动
streamlit run app.py
```

打开 http://localhost:8501，上传文档 → 构建索引 → 开始问答。

## 项目结构

```
rag-code-qa/
├── app.py                  # Streamlit 交互界面
├── ingest.py               # 文档摄入入口
├── rag_chain.py            # 兼容包装
├── download_models.py      # 模型预下载工具
├── prompts/                # YAML Prompt 模板
│   ├── rewrite.yaml        #   查询改写
│   ├── decompose.yaml      #   问题拆解
│   ├── qa.yaml             #   答案生成
│   └── validate.yaml       #   响应验证
├── src/
│   ├── config.py           # 全局配置
│   ├── ingest.py           # 文档加载与分块
│   ├── retrieve.py         # 混合检索器
│   ├── vectorstore.py      # FAISS / ChromaDB 抽象层
│   ├── qa.py               # QA 流水线
│   ├── validator.py        # 双模型响应验证
│   ├── eval.py             # RAGAS 评测
│   └── prompts.py          # Prompt 加载器
├── docs/                   # 文档目录（运行时）
└── data/                   # 索引目录（运行时）
```

## 配置说明

`.env` 中的主要配置项：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 必填 |
| `GENERATION_MODEL` | 生成答案的模型 | `deepseek-flash` |
| `VALIDATION_MODEL` | 验证答案的模型 | `deepseek-pro` |
| `ENABLE_VALIDATION` | 是否启用双模型验证 | `true` |
| `EMBEDDING_MODEL` | Embedding 模型（中文推荐 BAAI/bge-small-zh-v1.5） | `all-MiniLM-L6-v2` |
| `VECTOR_BACKEND` | 向量数据库（faiss / chroma） | `faiss` |

## 技术栈

LangChain / FAISS / ChromaDB / BM25 / jieba / BGE-Reranker / DeepSeek API / Streamlit / RAGAS
