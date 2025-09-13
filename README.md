
# RAG-based Medical FAQ Chatbot

## Overview

A Retrieval-Augmented Generation (RAG) chatbot that uses a small medical FAQ dataset and a FAISS vector store to retrieve relevant context and generate patient-friendly answers. It supports multiple interchangeable generation backends:

- OpenAI API (if you have API access / credits)
- Hugging Face model fallback (free, local)
- Ollama models (optional, local LLMs like Mistral 7B)

Supports multiple embeddings backends:

- HuggingFace SentenceTransformers
- OpenAI embeddings

## Quick 3-Command Start

Create virtual environment and install dependencies:

```bash


python -m venv venv

# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

Build FAISS index from your medical FAQs:

```bash
python src/ingest.py --csv data/faqs.csv --backend hf
```

- `--backend` can be hf or openai.
- `--openai_key` can be provided if using OpenAI embeddings.

**Note:** Currently, for faster testing, `ingest.py` uses `sample_df = df.head(100)` to build embeddings. Remove this line to embed the full dataset for production.

This generates:

- `data/faiss_index.bin`
- `data/meta.pkl`

Launch the chatbot UI:

```bash


streamlit run src/app.py
```

Open your browser at `http://localhost:8501`.

## Files

- `src/ingest.py` - build FAISS index from `data/faqs.csv`
- `src/embeddings.py` - embeddings wrapper (HF or OpenAI)
- `src/retriever.py` - vector search logic
- `src/generator.py` - generation (OpenAI, HF pipeline, or Ollama)
- `src/app.py` - Streamlit UI frontend

## FAQ Dataset

Prepare your medical FAQ CSV (`data/faqs.csv`) with columns:

```csv
question,answer
"What is LCM?","Lymphocytic Choriomeningitis is a viral infection..."
```

- Ensure no empty rows.
- Each row will be split into chunks of ~120 words for better retrieval.

## Configuration

Use the sidebar in Streamlit to configure:

- Generation backend: hf, openai, or ollama
- Embeddings backend: hf or openai
- OpenAI API Key: Required for OpenAI generation/embeddings
- Top-K retrieved chunks: Number of chunks to retrieve per query
- Ollama model name: Required if using Ollama (default: mistral:7b)

## How it Works

1. **Ingest:** Load medical FAQs and split long answers into chunks.
2. **Embed:** Convert text chunks into embeddings (HF or OpenAI).
3. **Index:** Store embeddings in FAISS for fast similarity search.
4. **Retrieve:** Given a user query, retrieve top-K relevant chunks.
5. **Generate:** Concatenate chunks and generate a patient-friendly answer using the selected AI backend.

**Default answer for bad context:** If retrieved context is irrelevant or empty, the generator provides a safe default answer suggesting consulting a doctor.

## Improving Answers for Unknown Questions

Sometimes questions may not be directly covered in the dataset, resulting in:

- Short or vague answers
- Irrelevant context retrieved

### Tips:

- Increase Top-K retrieval to provide more context.
- Combine context chunks into a single prompt to avoid losing info.
- Increase `max_tokens` in `generator.generate()` for longer answers.
- Summarize large context before generation to reduce token overload.
- Prompt engineering: Ask the model to provide detailed, patient-friendly answers and suggest consulting a doctor if unsure.
- Expand the dataset with more FAQs to cover broader topics.

## Customization

Change HuggingFace generation model:

```
generator = Generator(mode="hf", hf_model="google/flan-t5-large")
```

Change Ollama model:

```
generator = Generator(mode="ollama", ollama_model="mistral:7b")
```

- Adjust `max_tokens` for more or less detailed answers.
- Adjust chunk size in `ingest.py` if you want smaller/larger text pieces.
- Remove `sample_df = df.head(100)` for full dataset embeddings.

## Environment Variables

You can optionally set your OpenAI API key via environment variable:

```bash


export OPENAI_API_KEY="your_api_key"   # Linux/macOS
setx OPENAI_API_KEY "your_api_key"     # Windows
```

This avoids entering the key every time in the Streamlit sidebar.

## CUDA/GPU Support

Hugging Face models automatically use GPU if PyTorch detects CUDA.

Useful for faster generation with large models.

## Troubleshooting

- **Short/inaccurate answers:** Increase `max_tokens` or combine context chunks.
- **Ollama not found:** Ensure Ollama CLI is installed and in PATH.
- **OpenAI API errors:** Verify API key and network connectivity.
- **Index not loaded:** Click “Load/Refresh index” in the sidebar.
- **HF model not downloading:** Ensure `sentence-transformers` and `torch` are installed correctly.

## Limitations

- Small FAQ datasets may lead to missing context for uncommon questions.
- Default answers ensure safety but may not provide detailed info.
- This is a prototype, not a replacement for professional medical advice.

## Dependencies

- Python 3.9+
- streamlit
- faiss-cpu
- pandas
- sentence-transformers
- transformers
- torch (for HF models)
- openai (optional)
- Ollama CLI (optional, for local LLMs)

## License

MIT License. Free to use and modify.
