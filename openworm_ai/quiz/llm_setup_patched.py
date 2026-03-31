"""
Patched LLM setup for quiz evaluation - bypasses overly strict health checks.
"""

import os
import time
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings


def setup_llm_patched(model_name_full, logger):
    """Set up a chat model without strict health check assertions"""
    
    if model_name_full.lower().startswith("huggingface:"):
        _, model_name, provider = model_name_full.split(":")
        logger.debug(f"Using huggingface model: {model_name}")

        hf_token = os.environ.get("HF_TOKEN", None)
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set")

        logger.debug("Got HuggingFace Token")

        llm = HuggingFaceEndpoint(
            repo_id=f"{model_name}",
            provider="auto",
            max_new_tokens=4096,
            do_sample=False,
            repetition_penalty=1.03,
            task="conversational",
            huggingfacehub_api_token=hf_token,
        )

        model_var = ChatHuggingFace(llm=llm)

        # Try health check but don't fail if it times out
        try:
            result = model_var.invoke("Hello", config={"timeout": 5})
            logger.debug(f"Model health check passed: {result}")
        except Exception as e:
            logger.warning(f"Model health check failed (continuing anyway): {e}")
            # Don't fail - continue anyway
    
    else:
        # For non-HuggingFace models (OpenAI, Anthropic, Ollama)
        model_var = init_chat_model(
            model_name_full, configurable_fields=("temperature")
        )
        
        # Try health check
        try:
            result = model_var.invoke("Hello", config={"timeout": 10})
            logger.debug(f"Model health check passed")
        except Exception as e:
            logger.warning(f"Model health check failed (continuing anyway): {e}")

    logger.info(f"Using chat model: {model_name_full}")
    return model_var


def setup_embedding_patched(model_name_full, logger):
    """Set up embedding model - handles both 2-part and 3-part formats"""
    
    # Handle HuggingFace embeddings
    if model_name_full.lower().startswith("huggingface:"):
        parts = model_name_full.split(":")
        
        if len(parts) == 2:
            # Format: huggingface:BAAI/bge-large-en-v1.5
            _, model_name = parts
            provider = "auto"
        elif len(parts) == 3:
            # Format: huggingface:BAAI/bge-large-en-v1.5:auto
            _, model_name, provider = parts
        else:
            raise ValueError(f"Invalid model format: {model_name_full}")
        
        logger.debug(f"Using huggingface embedding model: {model_name}")

        hf_token = os.environ.get("HF_TOKEN", None)
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set")

        model_var = HuggingFaceEndpointEmbeddings(
            model=f"{model_name}",
            provider="auto",
            task="feature-extraction",
            huggingfacehub_api_token=hf_token,
        )
    else:
        # For other embedding models
        model_var = init_embeddings(model_name_full)

    logger.info(f"Using embedding model: {model_name_full}")
    return model_var