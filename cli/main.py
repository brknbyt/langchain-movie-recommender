import typer
from rich import print
from rich.panel import Panel
from typing_extensions import Annotated

import movie_recommender.config as config
from movie_recommender.llm import MovieRecommenderLLM


def question_loop(llm: MovieRecommenderLLM) -> None:
    """Run the main conversation loop with the movie recommender bot.

    Args:
        llm: The MovieRecommenderLLM instance to interact with.

    Raises:
        typer.Exit: When the user types 'exit' or 'quit'.
    """
    intro = (
        llm.introduce()
        + "\n\nType [italic yellow]exit[/italic yellow] to [red]quit[/red].\n"
    )
    print(Panel(intro, title="Cinephile Bot"))
    while True:
        user_input = input(">> ")
        if user_input.lower() in {"exit", "quit"}:
            print(Panel("Goodbye! :wave:", title="Cinephile Bot"))
            raise typer.Exit()
        response = llm.chat(user_input)
        print(Panel(response, title="Cinephile Bot"))


def main(
    debug: Annotated[
        bool,
        typer.Option(help="Shows verbose messages for easier debugging."),
    ] = False,
) -> None:
    """Main entry point for the movie recommender CLI application.

    Initializes the MovieRecommenderLLM with configuration from environment
    variables and starts the interactive question loop.

    Args:
        debug: Enable debug mode for verbose LangChain output.

    Raises:
        typer.Exit: If MODEL_NAME is not set in environment variables.
    """
    try:
        llm = MovieRecommenderLLM(model_name=config.MODEL_NAME, do_set_debug=debug)
    except ValueError:
        print(
            "[red]Error:[/red] No model name provided. Set MODEL_NAME in [italic].env[/]."
        )
        raise typer.Exit(code=1)
    question_loop(llm)


if __name__ == "__main__":
    typer.run(main)
