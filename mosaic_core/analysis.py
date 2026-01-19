#Move these functions: clean_text, get_embeddings, run_bertopic, run_umap, extract_topics from the app.py

"""
File: mosaic_core/analysis.py
Description: Core logic extracted from MOSAIC app.
             Pure Python implementation (no Streamlit dependencies).
"""

import pandas as pd
import numpy as np
import nltk
import json
import re
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# NLP / ML Imports
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from huggingface_hub import InferenceClient

# =====================================================================
# Constants (Copied from app.py)
# =====================================================================

SYSTEM_PROMPT = """You are an expert phenomenologist analysing first-person experiential reports or microphenomenological interviews.
Your task is to assign a concise label to a cluster of similar reports by identifying the
shared lived experiential structure or process they describe.
The label must:
1. Describe what changes in experience itself.
2. Capture the underlying experiential process.
3. Be concise and noun-phrase-like.
Constraints: Output ONLY the label (no explanation). 3–8 words.
"""

USER_TEMPLATE = """Here is a cluster of participant reports describing a specific phenomenon:

{documents}

Top keywords associated with this cluster:
{keywords}

Task: Return a single scientifically precise label (3–7 words). Output ONLY the label.
"""

# =====================================================================
# 1. Preprocessing & Embedding Logic
# =====================================================================

def load_embedding_model(model_name: str):
    print(f"Loading embedding model '{model_name}'...")
    return SentenceTransformer(model_name)

def _pick_text_column(df: pd.DataFrame) -> Optional[str]:
    """Helper to find the text column."""
    ACCEPTABLE_TEXT_COLUMNS = [
        "reflection_answer_english", "reflection_answer", "text", "report",
    ]
    for col in ACCEPTABLE_TEXT_COLUMNS:
        if col in df.columns:
            return col
    return None

def preprocess_and_embed(
    csv_path: str,
    model_name: str = "BAAI/bge-small-en-v1.5",
    text_col: Optional[str] = None,
    split_sentences: bool = True,
    min_words: int = 3,
    device: str = "cpu"
) -> Tuple[List[str], np.ndarray]:
    """
    Equivalent to 'generate_and_save_embeddings' but returns data instead of saving to disk.
    """
    # 1. Load CSV
    df = pd.read_csv(csv_path)

    # 2. Pick Column
    if text_col is None:
        text_col = _pick_text_column(df)
    
    if text_col is None or text_col not in df.columns:
        raise ValueError(f"Could not find a valid text column in {csv_path}")

    # 3. Clean NaN/Empty
    df.dropna(subset=[text_col], inplace=True)
    df[text_col] = df[text_col].astype(str)
    reports = [r for r in df[text_col] if r.strip()]

    # 4. Tokenize / Split
    docs = []
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
        
    if split_sentences:
        for r in reports:
            # Simple wrapper to avoid crashes
            sents = nltk.sent_tokenize(r)
            docs.extend(sents)
    else:
        docs = reports

    # 5. Filter min_words
    if min_words > 0:
        docs = [d for d in docs if len(d.split()) >= min_words]

    print(f"Preprocessing complete. {len(docs)} documents prepared.")

    # 6. Embed
    model = load_embedding_model(model_name)
    
    encode_device = "cpu"
    if device.lower() == "gpu":
        import torch
        if torch.cuda.is_available():
            encode_device = "cuda"
        elif torch.backends.mps.is_available():
            encode_device = "mps"

    print(f"Encoding on {encode_device}...")
    embeddings = model.encode(
        docs,
        show_progress_bar=True,
        batch_size=32,
        device=encode_device,
        convert_to_numpy=True
    )
    
    return docs, np.asarray(embeddings, dtype=np.float32)

# =====================================================================
# 2. Topic Modeling Logic
# =====================================================================

def run_topic_model(
    docs: List[str],
    embeddings: np.ndarray,
    config: Dict[str, Any]
):
    """
    Equivalent to 'perform_topic_modeling'.
    Config expects keys: umap_params, hdbscan_params, vectorizer_params, bt_params
    """
    # Unpack config (with defaults matching your app)
    umap_params = config.get("umap_params", {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0})
    hdbscan_params = config.get("hdbscan_params", {"min_cluster_size": 10, "min_samples": 5})
    vec_params = config.get("vectorizer_params", {})
    bt_params = config.get("bt_params", {"nr_topics": "auto", "top_n_words": 10})
    
    # Handle ngram_range tuple conversion
    if "ngram_range" in vec_params and isinstance(vec_params["ngram_range"], list):
        vec_params["ngram_range"] = tuple(vec_params["ngram_range"])

    # Instantiate models
    umap_model = UMAP(random_state=42, metric="cosine", **umap_params)
    hdbscan_model = HDBSCAN(metric="euclidean", prediction_data=True, **hdbscan_params)
    
    vectorizer_model = None
    if config.get("use_vectorizer", True):
        vectorizer_model = CountVectorizer(**vec_params)

    nr_topics = bt_params.get("nr_topics", "auto")
    if nr_topics != "auto":
        nr_topics = int(nr_topics)

    # Run BERTopic
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=bt_params.get("top_n_words", 10),
        nr_topics=nr_topics,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(docs, embeddings)
    
    # Calculate UMAP reduction for visualization (2D)
    reduced_2d = UMAP(
        n_neighbors=15, n_components=2, min_dist=0.0, metric="cosine", random_state=42
    ).fit_transform(embeddings)

    return topic_model, reduced_2d, topics

# =====================================================================
# 3. LLM Logic
# =====================================================================

def _clean_label(x: str) -> str:
    """Helper to clean LLM output"""
    x = (x or "").strip()
    x = x.splitlines()[0].strip()
    x = x.strip(' "\'`')
    return x or "Unlabelled"

def generate_llm_labels(
    topic_model: BERTopic,
    hf_token: str,
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    max_topics: int = 40,
    max_docs_per_topic: int = 8
) -> Dict[int, str]:
    """
    Equivalent to 'generate_labels_via_chat_completion' but without Streamlit progress bars.
    """
    client = InferenceClient(model=model_id, token=hf_token)
    
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info.Topic != -1].head(max_topics)
    
    labels = {}
    print(f"Generating labels for {len(topic_info)} topics...")

    for tid in topic_info.Topic.tolist():
        # Get keywords
        words = topic_model.get_topic(tid) or []
        keywords = ", ".join([w for (w, _) in words[:10]])

        # Get docs
        reps = (topic_model.get_representative_docs(tid) or [])[:max_docs_per_topic]
        docs_block = "\n".join([f"- {r}" for r in reps]) if reps else "(No docs)"

        # Prompt
        user_prompt = USER_TEMPLATE.format(documents=docs_block, keywords=keywords)

        try:
            out = client.chat_completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=24,
                temperature=0.2
            )
            raw = out.choices[0].message.content
            labels[int(tid)] = _clean_label(raw)
        except Exception as e:
            print(f"Error on topic {tid}: {e}")
            labels[int(tid)] = f"Topic {tid} (Error)"
            
    return labels