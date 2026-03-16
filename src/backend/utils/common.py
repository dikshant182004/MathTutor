import os 
from dotenv import load_dotenv

load_dotenv()

def _get_secret(key: str, default: str = "") -> str:
    """
    Read a secret from st.secrets (Streamlit Cloud) first,
    then fall back to os.getenv / .env (local development).
    This lets the same code run unchanged in both environments.
    """
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)

