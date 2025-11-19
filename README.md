# Movie Recommender CLI Bot

A conversational CLI bot powered by Langchain and Anthropic's Claude that helps users discover their next favorite movie through intelligent questioning and personalized recommendations.

## Features

- Interactive conversational interface for movie recommendations
- Built with Langchain for robust LLM integration
- Debug mode for development and troubleshooting
- Type-safe Python implementation with modern tooling

## Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic API key (or other provider, but requires adding the env var in .env)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/langchain-movie-recommender.git
cd langchain-movie-recommender
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up your environment variables:
```bash
cp .env.example .env
```

4. Edit `.env` and add your Anthropic API key:
```
ANTHROPIC_API_KEY="your-api-key-here"
MODEL_NAME="claude-sonnet-4-5-20250929"
```

## Usage

### Run the CLI

```bash
uv run movie-recommender
```

### Run with debug mode enabled

```bash
uv run movie-recommender-debug
```

Or pass the debug flag directly:

```bash
uv run movie-recommender --debug
```

### Interactive Session

Once started, the bot will greet you and begin asking questions to understand your movie preferences. Type your responses naturally and the bot will recommend movies based on your answers.

To exit the session, type `exit` or `quit`.

## Development

### Running Tests

Run all tests:
```bash
uv run pytest
```

Run tests with verbose output:
```bash
uv run pytest -v
```

### Code Formatting

This project uses Ruff for linting and formatting:
```bash
uv run ruff check .
uv run ruff format .
```

### Project Structure

```
langchain-movie-recommender/
├── src/
│   └── movie_recommender/
│       ├── __init__.py
│       ├── cli.py          # CLI interface and entry points
│       └── llm.py          # LLM integration and logic
├── tests/
│   ├── conftest.py
│   └── test_llm.py
├── .env.example            # Example environment configuration
├── pyproject.toml          # Project dependencies and configuration
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.


