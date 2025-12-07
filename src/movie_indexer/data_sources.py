import os
from typing import Any

import kagglehub
import pandas as pd
from langchain_community.document_loaders.dataframe import DataFrameLoader
from langchain_core.document_loaders import BaseLoader


class KaggleCSVDataSource:
    """Data source that loads movie data from a Kaggle dataset CSV file.

    This implementation downloads a Kaggle dataset, loads a specific CSV file,
    and configures a DataFrameLoader for it.
    """

    def __init__(
        self,
        dataset_handle: str,
        csv_filename: str,
        content_column: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the Kaggle CSV data source.

        Args:
            dataset_handle: Kaggle dataset identifier (e.g., "user/dataset-name").
            csv_filename: Name of the CSV file within the dataset.
            content_column: Column name to use as document content.
            **kwargs: Additional keyword arguments (e.g., drop_na).
        """
        self._dataset_handle = dataset_handle
        self._csv_filename = csv_filename
        self._content_column = content_column
        self._drop_na = kwargs.get("drop_na", True)

    def get_loader(self) -> BaseLoader:
        """Download the Kaggle dataset and return a configured DataFrameLoader.

        Returns:
            BaseLoader: A DataFrameLoader configured with the CSV data.
        """
        data_dir = kagglehub.dataset_download(self._dataset_handle)

        csv_path = os.path.join(data_dir, self._csv_filename)
        df = pd.read_csv(csv_path)

        if self._drop_na:
            df = df.dropna()

        return DataFrameLoader(df, page_content_column=self._content_column)

    @classmethod
    def from_env(cls) -> "KaggleCSVDataSource":
        """Factory method to create a KaggleCSVDataSource from environment variables.

        Expected environment variables:
            - KAGGLE_DATASET_HANDLE: Kaggle dataset identifier
            - CSV_FILENAME: Name of the CSV file (default: TMDB_all_movies.csv)
            - CSV_CONTENT_COLUMN: Column to use as content (default: overview)

        Returns:
            KaggleCSVDataSource: Configured instance from environment variables.
        """
        return cls(
            dataset_handle=os.getenv("KAGGLE_DATASET_HANDLE"),
            csv_filename=os.getenv("CSV_FILENAME", "TMDB_all_movies.csv"),
            content_column=os.getenv("CSV_CONTENT_COLUMN", "overview"),
            drop_na=True,
        )
