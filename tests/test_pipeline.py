import pytest
import numpy as np
from mosaic_core.analysis import preprocess_and_embed, run_topic_model

def test_full_pipeline_dummy():
    # create dummy data
    docs = [
        "I saw a bright light.", "The light was blinding and white.",
        "I felt a presence.", "The presence was comforting.",
        "Patterns emerged in the visual field.", "Geometric patterns were everywhere."
    ] * 5  # duplicate to have enough data for UMAP
    
    # mock Embeddings (rdm floats to avoid downloading models during test)
    # shape: (30 docs, 384 dimensions)
    embeddings = np.random.rand(len(docs), 384).astype(np.float32)
    
    # define config
    config = {
        "umap_params": {"n_neighbors": 2, "n_components": 2, "min_dist": 0.0},
        "hdbscan_params": {"min_cluster_size": 2, "min_samples": 1},
        "bt_params": {"nr_topics": 2, "top_n_words": 3},
        "vectorizer_params": {"stop_words": "english"}
    }
    
    # run Topic Model
    topic_model, reduced, topics = run_topic_model(docs, embeddings, config)
    
    # assertions (did it work?)
    assert len(topics) == len(docs)
    assert reduced.shape == (len(docs), 2)
    # check that we got a valid topic model object
    assert hasattr(topic_model, "get_topic_info")