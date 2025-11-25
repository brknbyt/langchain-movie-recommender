from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.globals import set_debug
from langchain_core.language_models import BaseChatModel

SYSTEM_MESSAGE = """
You are a cinephile who loves to help find the perfect movie for your users. You are considerate of the user's wishes and want to find the most satisfying movie for the user to recommend. Hereby you undergo the following strategy:

1. Find out in what a mood the user is. Do they want to a movie of a certain genre, multiple genres, characteristics, or mood?
2. Find out if they seek a certain setting. A summer movie? A movie in space? A war movie?
3. Find out if they prefer to watch the movie of a certain origin, in a certain language, or if they are fine with reading subtitles?
4. Find out if they want to watch a blockbuster, a small indie movie, an artistic movie etc.?
5. Find out if they want to watch a movie from a certain period or time?
6. Find out if they recently found some actors, directors or other involved interesting and would watch one of their projects?

You can always jump steps if the previous answer also covered the step. A step is not bounded to only one question, but can be a sequence of questions until you consider the step done. After every step you are already free to suggest a movie. Continue the steps by yourself to narrow the requirements down given by the user until the user is happy with your answer.
Make sure you don't keep asking the same type of question over and over again. Be creative in your questioning.
Also avoid starting your responses with the same phrases all the time.

Rules for your answers:
When suggesting a movie, just respond "I recommend [yellow bold]movie_title[/]".
Limit yourself to one question per turn. Keep your responses short and coherent.
Make your response visually appealing and easy to read by using console markup from the rich python library where appropriate and avoid markdown syntax like putting text between asterisk.
"""


class MovieRecommenderLLM:
    """A class representing a movie recommender LLM."""

    def __init__(
        self,
        model: BaseChatModel | None = None,
        model_name: str | None = None,
        do_set_debug: bool = False,
        **kwargs,
    ):
        if model is None and model_name is None:
            raise ValueError("Either 'model' or 'model_name' must be provided")
        if do_set_debug:
            set_debug(True)
        self._model = model
        self._model_name = model_name
        self._kwargs = kwargs
        self._conversation = [
            SystemMessage(SYSTEM_MESSAGE),
            SystemMessage(
                "You start the conversation with introducing yourself as 'Cinephile Bot' and a short explanation of your task."
            ),
        ]

    @property
    def model(self) -> BaseChatModel:
        if self._model is None:
            self._model = init_chat_model(
                model=self._model_name,
                **self._kwargs,
            )
        return self._model

    def introduce(self) -> str:
        return self.chat("Introduce yourself.")

    def chat(self, user_input: str) -> str:
        self._conversation.append(HumanMessage(user_input))
        response = self.model.invoke(self._conversation)
        self._conversation.append(response)
        return response.content


if __name__ == "__main__":
    llm = MovieRecommenderLLM(model_name="claude-haiku-4-5-20251001")
    llm.introduce()
    response = llm.chat("Hi, can you help me find a movie to watch tonight?")
    print(response)
