import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatPerplexity
# from langchain.chains import RetrievalQA
# from langchain import hub

# Self Query Retriever
# from langchain.retrievers import SelfQueryRetriever
# from langchain.embeddings import OpenAIEmbeddings
# from langchain_upstage import UpstageEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain.schema import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

##########################
# 문서 생성
# documents = [
#     Document(page_content="이 문서는 과학에 관한 내용입니다.", metadata={"title": "과학", "year":2025}),
#     Document(page_content="이 문서는 예술에 관한 내용입니다.", metadata={"title": "예술", "year":2024}),
# ]

# 메타데이터 정의
# metadata_field_info = [
#     AttributeInfo(
#         name="title",
#         description="The category of the cosmetic product, One of ['과학', '예술']",
#         type="string",
#     ),
#     AttributeInfo(
#         name="year",
#         description="The year the cosmetic product was released",
#         type="integer",
#     ),    
# ]
##########################

def get_retriever():    
    # 데이터를 처음 저장할 때 - 문서로드 및 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )

    loader = Docx2txtLoader('./tax_with_markdown.docx')
    document_list = loader.load_and_split(text_splitter=text_splitter) 

    # 임베딩 및 백터스토어 생성
    # embedding = OpenAIEmbeddings()
    # embedding = UpstageEmbeddings(model='embedding-query')
    embedding = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs = {'device': 'cpu'}, # 모델이 CPU에서 실행되도록 설정. GPU를 사용할 수 있는 환경이라면 'cuda'로 설정할 수도 있음
        encode_kwargs = {'normalize_embeddings': True}, # 임베딩 정규화. 모든 벡터가 같은 범위의 값을 갖도록 함. 유사도 계산 시 일관성을 높여줌
    )         
    database = FAISS.from_documents(documents=document_list, embedding=embedding)
    # database = Chroma.from_documents(documents=document_list, embedding=embedding, collection_name='chroma-tax', persist_directory="./chroma")
    
    
    # 이미 저장된 데이터를 사용할 때
    # database = Chroma(collection_name='chroma-tax',  persist_directory="./chroma", embedding_function=embedding)

    retriever = database.as_retriever(search_kwargs={'k': 3})

    return retriever
 

from config import answer_examples

load_dotenv()

pplx_api_key = os.getenv('PPLX_API_KEY')

store = {} 

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_llm():
    llm = ChatPerplexity(
        temperature=0, pplx_api_key=pplx_api_key, model="sonar-pro"
    )
    return llm

def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 → 거주자"]

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()

    return dictionary_chain

def get_history_retriever():

    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def get_rag_chain():
    llm = get_llm()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )    

    history_aware_retriever = get_history_retriever()

    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ]
    ) 

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever , question_answer_chain)       

    conversational_rag_chain  = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key ="input",
        history_messages_key ="chat_history",
        output_messages_key="answer"
    ).pick('answer')

    # return rag_chain
    return conversational_rag_chain    

def get_ai_message(user_message):
    llm = get_llm()

    dictionary_chain = get_dictionary_chain()
    rag_chain  = get_rag_chain()

    tax_chain = {"input": dictionary_chain} | rag_chain        

    # invoke 일때 any→ stream으로 바꾸면 Iterator로 바뀜   invoke 할때는 spinner를 보고 있어서 오래 걸리는 느낌.
    ai_response = tax_chain.stream(
    # ai_message = tax_chain.invoke(
        {
            "question": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response
