---
title: MOSAICapp
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# MOSAIC Topic Dashboard

A Streamlit app for BERTopic-based topic modelling with sentence-transformers embeddings.
**No data bundled** — upload CSV with one text column (any of: `reflection_answer_english`, `reflection_answer`, `text`, `report`).

## Lite Version (Free Hardware)

This Hugging Face Space runs the **`lite` version** of the app.

To make it run on free "CPU basic" hardware, the **LLM-based topic labeling feature has been disabled**. The app will use BERTopic's default keyword-based labels instead.

For the full, original version with LLM features (which requires paid GPU hardware), please see the `main` branch of the [original GitHub repository](https://github.com/romybeaute/MOSAICapp).

## Run Locally (Full Version)

To run the full version on your local machine:

```bash
# Clone the main branch
git clone [https://github.com/romybeaute/MOSAICapp.git](https://github.com/romybeaute/MOSAICapp.git)
cd MOSAICapp

# Install requirements
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Run the app
streamlit run app.py