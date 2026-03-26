from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool


def make_movie_recommendation_tool(store) -> BaseTool:
    """Factory function to create a movie recommendation tool.

    This function takes a vector store as input and returns a tool that can be used
    by the agent to perform movie recommendations based on user queries.

    Args:
        store: The vector store instance to use for similarity search.

    Returns:
        tool: A LangChain tool that performs movie recommendations.
    """

    @tool()
    def movie_recommendation(query: str) -> list[Document]:
        """Tool to recommend movies based on a user query.

        Searches the vector store for movies that match the user's query
        using semantic similarity search.

        Args:
            query: The user's search query describing their movie preferences.

        Returns:
            list[Document]: A list of Document objects representing matching movies.
        """
        res = store.similarity_search(query)
        return res

    return movie_recommendation
