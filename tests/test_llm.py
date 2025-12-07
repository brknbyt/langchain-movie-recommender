"""Tests for the MovieRecommenderLLM class."""

from movie_recommender.llm import MovieRecommenderLLM
from tests.conftest import FakeChatModelWithTools


def test_create_llm_with_model(fake_model: FakeChatModelWithTools) -> None:
    """Test creating MovieRecommenderLLM with a pre-initialized model."""
    llm = MovieRecommenderLLM(model=fake_model)
    assert llm.model == fake_model


def test_create_llm_with_model_name() -> None:
    """Test creating MovieRecommenderLLM with a model name string."""
    llm = MovieRecommenderLLM(model_name="claude-haiku-4-5-20251001")
    assert llm.model is not None
    assert llm.model.model == "claude-haiku-4-5-20251001"


def test_chat_with_fake_model(fake_model: FakeChatModelWithTools) -> None:
    """Test the chat method with a fake model."""
    llm = MovieRecommenderLLM(model=fake_model)
    assert len(llm._conversation) == 2  # Initial messages
    response = llm.chat("Hi, can you help me find a movie to watch tonight?")
    assert response == "What about **The Holy Mountain**?"
    assert len(llm._conversation) == 4  # Initial + user + response
