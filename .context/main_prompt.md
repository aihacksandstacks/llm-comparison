You are an expert AI developer assistant. 
Your task is to generate a scaffold for an LLM comparison tool with the following requirements:

1. **Workflow orchestration**  
   - Use llama_index to ingest data (web pages, PDFs, code) into a vector store and execute retrieval-augmented queries against multiple LLMs  [oai_citation:0‡LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/?utm_source=chatgpt.com).
2. **Embedding layer**  
   - Integrate Nomic Atlas embeddings for text and code, with flexibility to swap embedding providers  [oai_citation:1‡Atlas | Nomic Atlas Documentation](https://docs.nomic.ai/atlas/embeddings-and-retrieval/generate-embeddings?utm_source=chatgpt.com) [oai_citation:2‡Introduction | 🦜️🔗 LangChain](https://python.langchain.com/docs/integrations/text_embedding/nomic/?utm_source=chatgpt.com).
3. **Local model serving**  
   - Serve models via Ollama on the user’s machine, with Docker-compose config and GPU detection  [oai_citation:3‡Medium](https://medium.com/cyberark-engineering/how-to-run-llms-locally-with-ollama-cb00fa55d5de?utm_source=chatgpt.com) [oai_citation:4‡Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1iobld4/is_ollama_good_enough_for_a_local_llm_server_for/?utm_source=chatgpt.com).
4. **Evaluation framework**  
   - Instrument Comet ML Opik to log prompts, responses, and evaluation metrics; enable automated benchmarking and annotation  [oai_citation:5‡Comet](https://www.comet.com/site/products/opik/?utm_source=chatgpt.com) [oai_citation:6‡GitHub](https://github.com/comet-ml/opik?utm_source=chatgpt.com).
5. **Web crawling**  
   - Incorporate Crawl4AI to crawl and preprocess websites into the vector store for RAG  [oai_citation:7‡GitHub](https://github.com/unclecode/crawl4ai?utm_source=chatgpt.com) [oai_citation:8‡Generative AI with Local LLM](https://quickstartgenai.com/blog/web-crawling-for-rag-with-crawlai?utm_source=chatgpt.com).
6. **UI**  
   - Build an interactive Streamlit interface for selecting models, running tests, visualizing metrics, and comparing outputs  [oai_citation:9‡Streamlit Docs](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps?utm_source=chatgpt.com) [oai_citation:10‡GitHub](https://github.com/streamlit/llm-examples?utm_source=chatgpt.com).
7. **Extensibility**  
   - Structure code modularly to add new LLM backends, embedding providers, and evaluation metrics without rewriting core logic.
8. **Deliverables**  
   - Provide a directory scaffold, Docker-compose files, Python modules with docstrings, unit tests, and a README with setup and usage.
   
Begin by generating the project layout, then implement each module with example code stubs. End with instructions to run the full stack locally.