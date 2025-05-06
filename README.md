# 설치

## 파일 설명

- chat.py는 챗팅 기본 파일일
- llm.py는 perplexity 연결 하는 파일
  - llm_pplx_huggingembedding.py는 embedding 모델은 HuggingFaceEmbeddings을 사용함
  - llm_pplx_upstageembedding.py는 embedding 모델은 UpstageEmbedding을 사용함

```shell
$ uv init pyenv-st-chat
$ cd pyenv-st-chat
$ uv venv --python 3.11.9
$ .venv\Scripts\activate
# streamlit 설치
$ uv add streamlit

# OpenAI 설치
$ pip install langchain langchain-core langchain-community openai python-dotenv docx2txt langchain-text-splitters tiktoken #강의
or
$ uv add langchain langchain-core langchain-community openai python-dotenv docx2txt langchain-text-splitters tiktoken

# Upstage 설치
$ pip install langchain langchain-core langchain-community langchain-upstage python-dotenv docx2txt langchain-text-splitters tiktoken #강의
or
$ uv add langchain langchain-core langchain-community langchain-upstage python-dotenv docx2txt langchain-text-splitters tiktoken

# Huggingface 모델델 설치
$ pip install langchain langchain-core langchain-community sentence-transformers python-dotenv docx2txt langchain-text-splitters tiktoken faiss-cpu #강의
or
$ uv add langchain langchain-core langchain-community sentence-transformers python-dotenv docx2txt langchain-text-splitters tiktoken faiss-cpu
 
# streamlit 실행
$ uv run streamlit run chat.py
```