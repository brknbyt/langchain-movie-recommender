import typer
from rich import print
from rich.panel import Panel

from movie_recommender.llm import MovieRecommenderLLM

INTRO = "Hi, I am a cinephile bot. I am here to help you find the perfect movie for today. We will go through a serious of questions until you are happy with my recommendation."


def question_loop():
    llm = MovieRecommenderLLM(
        model_name="claude-haiku-4-5-20251001", introduction_text=INTRO
    )
    while True:
        user_input = input(">> ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            raise typer.Exit()
        response = llm.chat(user_input)
        print(Panel(response, title="Cinephile Bot"))


def print_intro():
    print("Type [italic yellow]exit[/italic yellow] to [red]quit[/red].\n")
    print(Panel(INTRO, title="Cinephile Bot"))


def main():
    print_intro()
    question_loop()


if __name__ == "__main__":
    typer.run(main)
