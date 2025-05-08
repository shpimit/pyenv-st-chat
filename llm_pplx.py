import os
from dotenv import load_dotenv
# from langchain_community.chat_models import ChatPfrom langchain_perplexity import ChatPerplexity
from langchain_perplexity import ChatPerplexity
# from langchain.chains import RetrievalQA
# from langchain import hub


load_dotenv()

pplx_api_key = os.getenv('PPLX_API_KEY')


def get_ai_message(user_message):
    llm = ChatPerplexity(
        temperature=0, pplx_api_key=pplx_api_key, model="sonar-pro"
    )

    # messages = [("system", "You are a chatbot."), ("user", "Hello!")]
    ai_message = llm.invoke(user_message)     

    return ai_message
