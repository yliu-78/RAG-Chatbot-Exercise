import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import src.config as config


def get_llm():
    if config.LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0)


# llm = get_llm()



