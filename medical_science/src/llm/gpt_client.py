import os
from typing import Any
from dotenv import load_dotenv

# Reuse existing deps already required by project
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()


def get_chat_llm(temperature: float | None = None) -> Any:
    return AzureChatOpenAI(
        deployment_name=os.getenv("deployment_name"),
        model_name=os.getenv("model_name"),
        azure_endpoint=os.getenv("azure_endpoint"),
        openai_api_key=os.getenv("openai_api_key"),
        openai_api_version=os.getenv("openai_api_version"),
        temperature=temperature if temperature is not None else 0.3,
    )


def get_embedding_model() -> Any:
    return AzureOpenAIEmbeddings(
        deployment=os.getenv("embedding_deployment_name"),
        model=os.getenv("embedding_model_name"),
        azure_endpoint=os.getenv("embedding_azure_endpoint"),
        openai_api_key=os.getenv("embedding_openai_api_key"),
        openai_api_version=os.getenv("embedding_openai_api_version"),
    ) 