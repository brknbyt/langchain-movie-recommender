from typing import Any, Sequence

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool


class FakeChatModelWithTools(FakeListChatModel):
    """Fake chat model that supports tool binding for testing.

    Extends FakeListChatModel to implement bind_tools() method, which is
    required when creating LangGraph agents. This allows the fake model
    to work with create_agent() function.
    """

    def bind_tools(
        self, tools: Sequence[BaseTool], **kwargs: Any
    ) -> Runnable:
        """Mock bind_tools method to support agent creation.

        Args:
            tools: Sequence of tools to bind to the model.
            **kwargs: Additional keyword arguments.

        Returns:
            Runnable: Returns self to maintain the fake responses behavior.
        """
        # Return self to maintain the fake responses behavior
        # The agent will use this model which will return predefined responses
        return self


@pytest.fixture
def fake_model() -> FakeChatModelWithTools:
    """Create a fake chat model that supports tool binding.

    Returns:
        FakeChatModelWithTools: A fake model with predefined responses.
    """
    return FakeChatModelWithTools(responses=["What about **The Holy Mountain**?"])
