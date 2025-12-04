import os
from unittest.mock import patch

import pandas as pd

from movie_indexer.data_sources import KaggleCSVDataSource


def test_data_source_initialization():
    os.environ["KAGGLE_DATASET_HANDLE"] = "user/dataset-name"
    os.environ["CSV_FILENAME"] = "movies.csv"
    os.environ["CSV_CONTENT_COLUMN"] = "description"

    data_source = KaggleCSVDataSource.from_env()
    assert data_source._dataset_handle == "user/dataset-name"
    assert data_source._csv_filename == "movies.csv"
    assert data_source._content_column == "description"
    assert data_source._drop_na is True


@patch("movie_indexer.data_sources.kagglehub.dataset_download")
@patch("movie_indexer.data_sources.pd.read_csv")
def test_get_loader_mocked(mock_read_csv, mock_dataset_download):
    mock_dataset_download.return_value = "/fake/data/dir"
    mock_df = pd.DataFrame(
        {
            "title": ["Movie 1", "Movie 2", "Movie 3"],
            "overview": ["Description 1", "Description 2", None],
            "rating": [8.5, 7.0, 6.5],
        }
    )
    mock_read_csv.return_value = mock_df

    data_source = KaggleCSVDataSource(
        dataset_handle="user/dataset-name",
        csv_filename="movies.csv",
        content_column="overview",
        drop_na=True,
    )

    loader = data_source.get_loader()

    mock_dataset_download.assert_called_once_with("user/dataset-name")
    mock_read_csv.assert_called_once_with("/fake/data/dir/movies.csv")

    assert loader is not None

    documents = loader.load()
    assert len(documents) == 2
    assert documents[0].page_content == "Description 1"
    assert documents[1].page_content == "Description 2"
