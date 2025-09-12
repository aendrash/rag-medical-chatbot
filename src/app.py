# src/app.py
import streamlit as st
from retriever import Retriever
from generator import Generator
import os

st.set_page_config(page_title="RAG Medical FAQ Chatbot", layout="centered")
st.title("RAG-based Medical FAQ Chatbot")

# Configuration
MODE = st.sidebar.selectbox("Generation backend", ["hf", "openai", "ollama"])
EMBED_BACKEND = st.sidebar.selectbox("Embeddings backend", ["hf", "openai"])
OPENAI_KEY = st.sidebar.text_input("OpenAI API Key (optional)", type="password")

OLLAMA_MODEL = None
if MODE == "ollama":
    OLLAMA_MODEL = st.sidebar.text_input("Ollama model name", value="mistral:7b")

TOP_K = st.sidebar.slider("Top-K retrieved chunks", 1, 10, 4)
SIMILARITY_THRESHOLD = 0.4  # filter irrelevant context

# Load / refresh index
if st.button("Load/Refresh index"):
    st.info("Loading index... (this may take a few seconds)")
    retriever = Retriever(embed_backend=EMBED_BACKEND, openai_key=OPENAI_KEY or None)
    generator = Generator(
        mode=MODE,
        openai_api_key=OPENAI_KEY or None,
        ollama_model=OLLAMA_MODEL
    )
    st.session_state["retriever"] = retriever
    st.session_state["generator"] = generator
    st.success("Loaded!")

# Initialize if not loaded yet
if "retriever" not in st.session_state:
    try:
        st.session_state["retriever"] = Retriever(embed_backend=EMBED_BACKEND, openai_key=OPENAI_KEY or None)
        st.session_state["generator"] = Generator(
            mode=MODE,
            openai_api_key=OPENAI_KEY or None,
            ollama_model=OLLAMA_MODEL
        )
    except Exception as e:
        st.warning(f"Index or models not ready: {e}")

# User query
query = st.text_input("Ask a medical question:", "")
if st.button("Ask") and query.strip():
    retriever = st.session_state.get("retriever")
    generator = st.session_state.get("generator")
    if retriever is None or generator is None:
        st.error("Indexer / generator not loaded. Click 'Load/Refresh index' or check logs.")
    else:
        with st.spinner("Retrieving relevant documents..."):
            results = retriever.query(query, top_k=TOP_K)
            # Filter out low-similarity results
            results = [r for r in results if r["score"] > SIMILARITY_THRESHOLD]

        if not results:
            st.subheader("Answer")
            st.write("I am not sure about this. Please consult a medical professional.")
        else:
            st.subheader("Retrieved context (top results)")
            for r in results:
                st.markdown(f"**Score:** {r['score']:.3f} — **Source Question:** {r['source_question']}")
                st.write(r["text"])

            contexts = [r["text"] for r in results]

            # Combine all contexts into a single text for HF / Ollama
            if MODE in ["hf", "ollama"] and len(contexts) > 1:
                combined_context = "\n\n---\n\n".join(contexts)
                contexts = [combined_context]  # single combined context

            # Generate final answer
            with st.spinner("Generating answer..."):
                answer = generator.generate(
                    user_query=query,
                    contexts=contexts,
                    max_tokens=512  # increase token limit for detailed answers
                )

            st.subheader("Answer")
            st.write(answer)
            print("answer:", answer)
