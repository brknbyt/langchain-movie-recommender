import os

import typer
from dotenv import load_dotenv
from rich import print
from rich.panel import Panel

from movie_recommender.llm import MovieRecommenderLLM

load_dotenv()


def question_loop(llm: MovieRecommenderLLM) -> None:
    intro = (
        llm.introduce()
        + "\n\nType [italic yellow]exit[/italic yellow] to [red]quit[/red].\n"
    )
    print(Panel(intro, title="Cinephile Bot"))
    while True:
        user_input = input(">> ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            raise typer.Exit()
        response = llm.chat(user_input)
        print(Panel(response, title="Cinephile Bot"))


def main() -> None:
    model_name = os.getenv("MODEL_NAME")
    try:
        llm = MovieRecommenderLLM(model_name=model_name)
    except ValueError:
        print(
            "[red]Error:[/red] No model name provided. Set MODEL_NAME in [italic].env[/]."
        )
        raise typer.Exit(code=1)
    question_loop(llm)


if __name__ == "__main__":
    typer.run(main)
