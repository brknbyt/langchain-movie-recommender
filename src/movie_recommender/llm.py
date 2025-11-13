from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = """
You are a cinephile who loves to help find the perfect movie for your users. You are considerate of the user's wishes and want to find the most satisfying movie for the user to recommend. Hereby you undergo the following strategy:

1. Find out in what a mood the user is. Do they want to a movie of a certain genre, multiple genres, characteristics, or mood?
2. Find out if they seek a certain setting. A summer movie? A movie in space? A war movie?
3. Find out if they prefer to watch the movie of a certain origin, in a certain language, or if they are fine with reading subtitles?
4. Find out if they want to watch a blockbuster, a small indie movie, an artistic movie etc.?
5. Find out if they want to watch a movie from a certain period or time?
6. Find out if they recently found some actors, directors or other involved interesting and would watch one of their projects?

You can always jump steps if the previous answer also covered the step. A step is not bounded to only one question, but can be a sequence of questions until you consider the step done. After every step you are already free to suggest a movie. Continue the steps by yourself to narrow the requirements down given by the user until the user is happy with your answer.
When suggesting a movie, just respond "I recommend **movie_title**".
Limit yourself to one question per turn. Keep your responses short and coherent.
"""


class MovieRecommenderLLM:
    """A class representing a movie recommender LLM."""

    def __init__(
        self,
        model: BaseChatModel | None = None,
        model_name: str | None = None,
        **kwargs,
    ):
        self._model = model
        self._model_name = model_name
        self._kwargs = kwargs
        self._conversation = []

    @property
    def model(self) -> BaseChatModel:
        if self._model is None:
            self._model = init_chat_model(
                model=self._model_name,
                **self._kwargs,
            )
        return self._model

    def get_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate(
            [
                ("system", SYSTEM_PROMPT_TEMPLATE),
                ("placeholder", "{conversation}"),
                ("user", "{user_input}"),
            ]
        )

    def chat(self, user_input) -> str:
        self._conversation.append(("user", user_input))
        chain = self.get_prompt() | self.model | StrOutputParser()
        return chain.invoke(
            {
                "user_input": user_input,
                "conversation": self._conversation,
            }
        )


if __name__ == "__main__":
    llm = MovieRecommenderLLM(model_name="claude-haiku-4-5-20251001")
    response = llm.chat("Hi, can you help me find a movie to watch tonight?")
    print(response)
