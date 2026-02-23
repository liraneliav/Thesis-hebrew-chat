# Thesis-hebrew-chat
# Streamlit Thesis App (Hebrew)

This repository contains a Streamlit app for a user experiment.
It retrieves "opposite" comments using a topic-filter → NLI → optional GPT rerank pipeline, measures toxicity, and stores results in Firestore.

## What’s in this repo
- `chat_hebrew.py` – Streamlit UI + experiment flow
- `opposite_english_nli.py` – artifacts loader + NLI utilities
- `opposite_english_nli_gpt.py` – opposite retrieval pipeline (NLI + optional GPT)
- `toxicity.py` – toxicity scoring pipeline
- `firebase_store_english.py` – Firestore save helpers

## Local Setup

### 1) Install dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```
### 2) Create a file named .env in the project root
```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY=YOUR_AZURE_KEY
ENDPOINT_URL=https://YOUR-RESOURCE.openai.azure.com/

# Your code checks OPENAI_API_KEY exists; set any non-empty value if you only use Azure
OPENAI_API_KEY=placeholder

# Hugging Face token (only if your HF repo is private)
HF_TOKEN=YOUR_HF_TOKEN

# Firebase: path to service account json file (local dev)
FIREBASE_SERVICE_ACCOUNT=/absolute/path/to/firebase-service-account.json
```
### 3) Run the app
```bash
streamlit run chat_hebrew.py
```
