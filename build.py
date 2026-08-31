# build_index.py

import os
import re
from pathlib import Path
from typing import Callable
import torch
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from adaptive_chunking import (
    AdaptiveChunkConfig,
    adaptive_split_documents,
    approximate_token_count,
)
# --- 配置 ---
# 在 Mac、Windows 或 VS Code 中都以项目目录为基准。
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
os.environ.setdefault("HF_HOME", str(BASE_DIR / ".cache" / "huggingface"))

# 文件和模型路径配置
PDF_FOLDER_PATH = BASE_DIR / "金融数据集-报表"
EMBEDDING_MODEL_NAME_OR_PATH = os.getenv(
    "EMBEDDING_MODEL_NAME_OR_PATH",
    "BAAI/bge-small-zh-v1.5",
)
FAISS_DB_PATH = BASE_DIR / "faiss_index"
METADATA_FILE_NAME = "documents_metadata.json"

# 语雀 1.5.4：按解析元素累计 token，超过 max_tokens 时结束当前块。
ADAPTIVE_CHUNK_CONFIG = AdaptiveChunkConfig(
    max_tokens=int(os.getenv("CHUNK_MAX_TOKENS", "448")),
    overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "96")),
)

# 计算设备
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

def company_name_from_filename(filename: str) -> str:
    """把 ``11_滨江消费品有限公司_report.pdf`` 清洗为公司名称。"""
    stem = Path(filename).stem
    stem = re.sub(r"^\d+_", "", stem)
    stem = re.sub(r"_report$", "", stem, flags=re.IGNORECASE)
    return stem


def normalize_text(text: str) -> str:
    """用于识别完全重复的文本块，同时保留原文用于展示。"""
    return re.sub(r"\s+", "", text).strip()


def get_token_counter(model_name_or_path: str) -> Callable[[str], int]:
    """加载与 Embedding 相同的 tokenizer，避免字符数与真实 token 数偏差。"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception as exc:
        print(f"Tokenizer 加载失败，将使用保守估算: {exc}")
        return approximate_token_count

    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def load_and_split_pdf(
    folder_path: str,
    company_name: str = None,
    count_tokens: Callable[[str], int] = approximate_token_count,
) -> list[Document]:
    """加载 PDF 文档并进行文本切分。"""
    all_chunks = []

    seen_chunks: set[tuple[str, str]] = set()

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            name = company_name if company_name else company_name_from_filename(file)

            print(f"正在加载 PDF 文件: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"原始文档页数: {len(documents)}")

            chunks = adaptive_split_documents(
                documents,
                ADAPTIVE_CHUNK_CONFIG,
                count_tokens=count_tokens,
            )
            unique_chunks = []
            for chunk in chunks:
                chunk.metadata["company"] = name
                chunk.metadata["source_file"] = file
                fingerprint = (name, normalize_text(chunk.page_content))
                if not fingerprint[1] or fingerprint in seen_chunks:
                    continue
                seen_chunks.add(fingerprint)
                unique_chunks.append(chunk)
            print(
                "文档自适应切分完成，"
                f"上限 {ADAPTIVE_CHUNK_CONFIG.max_tokens} tokens，"
                f"生成 {len(chunks)} 个文本块，"
                f"去重后保留 {len(unique_chunks)} 个。"
            )

            all_chunks.extend(unique_chunks)

    return all_chunks

def get_embeddings_model(model_name_or_path: str) -> HuggingFaceEmbeddings:
    """获取嵌入模型。"""
    print(f"正在加载嵌入模型: {model_name_or_path} (device: {DEVICE})")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name_or_path,
        model_kwargs={'device': DEVICE}
    )
    print("嵌入模型加载完成。")
    return embeddings

def create_and_save_faiss_db(chunks: list[Document], embeddings_model: HuggingFaceEmbeddings, db_path: str):
    """创建 FAISS 向量数据库并保存到本地。"""
    print("正在创建 FAISS 向量数据库...")
    faiss_db = FAISS.from_documents(chunks, embeddings_model)
    print(f"FAISS 向量数据库创建完成。正在保存到: {db_path}")
    faiss_db.save_local(db_path)
    print("FAISS 向量数据库保存成功。")

def create_and_save_metadata(chunks: list[Document], output_dir: str, metadata_file_name: str):
    """创建并保存文档块的元数据。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metadata_list = [{"chunk_id": i, "content": chunk.page_content, "metadata": chunk.metadata} for i, chunk in enumerate(chunks)]
    metadata_file_path = os.path.join(output_dir, metadata_file_name)
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)
    print(f"元数据已保存到：{metadata_file_path}")
def save_documents_for_bm25(chunks: list[Document], output_dir: str, file_name: str = "bm25_documents.json"):
    """将结构化文档块以 BM25 可读形式保存"""
    doc_list = []
    for doc in chunks:
        doc_list.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as f:
        json.dump(doc_list, f, ensure_ascii=False, indent=4)
    print(f"BM25 文档已保存到：{file_name}")
    
if __name__ == "__main__":
    if not os.path.exists(PDF_FOLDER_PATH):
        print(f"错误：文件夹未找到，请确保 '{PDF_FOLDER_PATH}' 存在。")
    else:
        try:
            # 1. 每份 PDF 只处理一次，并使用文件名作为公司名。
            token_counter = get_token_counter(EMBEDDING_MODEL_NAME_OR_PATH)
            all_chunks = load_and_split_pdf(
                PDF_FOLDER_PATH,
                count_tokens=token_counter,
            )
            # 2. 初始化嵌入模型
            embeddings_model = get_embeddings_model(EMBEDDING_MODEL_NAME_OR_PATH)
            # 3. 创建并保存 FAISS 向量数据库
            create_and_save_faiss_db(all_chunks, embeddings_model, FAISS_DB_PATH)
            # 4. 创建并保存元数据
            create_and_save_metadata(all_chunks, FAISS_DB_PATH, METADATA_FILE_NAME)
            # 5. 保存 BM25 可读的结构化文档（供 FastAPI 使用）
            save_documents_for_bm25(all_chunks, FAISS_DB_PATH)
            print("\n索引和元数据和bm25构建完成！")
        except Exception as e:
            print(f"\n构建过程中发生错误: {e}")
            
