import typer
from rich import print
from rich.panel import Panel

from movie_recommender.llm import MovieRecommenderLLM


def question_loop():
    llm = MovieRecommenderLLM(model_name="claude-haiku-4-5-20251001")
    print(Panel(llm.introduce(), title="Cinephile Bot"))
    while True:
        user_input = input(">> ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            raise typer.Exit()
        response = llm.chat(user_input)
        print(Panel(response, title="Cinephile Bot"))


def print_intro():
    print("Type [italic yellow]exit[/italic yellow] to [red]quit[/red].\n")


def main():
    question_loop()


if __name__ == "__main__":
    typer.run(main)
