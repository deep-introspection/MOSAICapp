# MOSAICapp Test Suite

This directory contains the automated tests for the MOSAICapp software.

## File Structure

### 1. `conftest.py`
Contains **fixtures** (setup code) shared across all tests. It generates temporary CSV files and dummy datasets so tests can run without external data dependencies.

### 2. `test_core_functions.py` (Fast)
Unit tests for the `mosaic_core` utility functions. These run instantly.
- **Slugify:** Checks filename sanitisation.
- **CacheIO:** Verifies that topic labels can be saved/loaded from JSON.
- **Device Resolution:** Ensures the app correctly selects CPU/GPU/MPS devices.

### 3. `test_integration.py` (Slow)
Integration tests that run the actual ML pipeline.
- **Embeddings:** Loads the `sentence-transformers` model and checks vector output.
- **Topic Modelling:** Runs a minimal BERTopic pipeline (UMAP + HDBSCAN) on dummy data.
- **LLM labeling:** (Optional) Tests the HuggingFace API connection if `HF_TOKEN` is present in the environment.

## How to Run

**Run everything:**
```bash
pytest tests/ -v
```

**Run only fast tests:**
```bash
pytest tests/test_core_functions.py -v
```
