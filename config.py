import os
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, SecretStr
load_dotenv(find_dotenv(), override=True)

class config(BaseModel):
    CLIP_MODEL_NAME: str
    CLIP_EMBEDDING_DIM: int
    MODAL_VLM_URL: SecretStr
    LANCEDB_URI: SecretStr
    HF_API_KEY: SecretStr
    MODEL_API_KEY: SecretStr
    NEBIUS_API_KEY: SecretStr
    CLIP_EMBEDDING_URL: SecretStr
    OPENAI_API_KEY: SecretStr

def load_config():
    return config(
        CLIP_MODEL_NAME = "openai/clip-vit-base-patch32",
        CLIP_EMBEDDING_DIM = 512,
        MODAL_VLM_URL = os.getenv("MODAL_VLM_URL"),
        LANCEDB_URI = os.getenv("LANCEDB_URI"),
        MODEL_API_KEY = os.getenv("MODAL_API_KEY"),
        HF_API_KEY = os.getenv("HF_API_KEY"),
        NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY"),
        CLIP_EMBEDDING_URL = os.getenv("MODAL_EMBEDDING_SERVER"),
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    )