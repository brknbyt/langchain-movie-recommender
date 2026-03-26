import os

from dotenv import load_dotenv

load_dotenv()


def required_env_var(name: str) -> str:
    """Helper function to get a required environment variable.

    Args:
        name: The name of the environment variable to retrieve.

    Returns:
        The value of the environment variable.

    Raises:
        ValueError: If the environment variable is not set.
    """
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"{name} is not set in environment variables.")
    return value


MODEL_NAME: str = os.getenv("MODEL_NAME", "claude-sonnet-4-6")
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_STORE_NAME: str = os.getenv("VECTOR_STORE", "in_memory")

if VECTOR_STORE_NAME == "pgvectorstore":
    PG_USER = required_env_var("POSTGRES_USER")
    PG_PASSWORD = required_env_var("POSTGRES_PASSWORD")
    PG_HOST = required_env_var("POSTGRES_HOST")
    PG_PORT = required_env_var("POSTGRES_PORT")
    PG_DB = required_env_var("POSTGRES_DB")
    TABLE_NAME = required_env_var("TABLE_NAME")
    VECTOR_SIZE = int(required_env_var("VECTOR_SIZE"))


KAGGLE_DATASET_PATH: str = os.getenv("KAGGLE_DATASET_PATH", "user/dataset-name")
CSV_FILENAME: str = os.getenv("CSV_FILENAME", "TMDB_all_movies.csv")
CSV_CONTENT_COLUMN: str = os.getenv("CSV_CONTENT_COLUMN", "overview")
