import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel


@pytest.fixture
def fake_model():
    return FakeListChatModel(responses=["What about **The Holy Mountain**?"])
