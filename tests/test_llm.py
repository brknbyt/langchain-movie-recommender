from movie_recommender.llm import MovieRecommenderLLM


def test_create_llm_with_model(fake_model):
    llm = MovieRecommenderLLM(model=fake_model)
    assert llm.model == fake_model


def test_create_llm_with_model_name():
    llm = MovieRecommenderLLM(model_name="claude-haiku-4-5-20251001")
    assert llm.model is not None
    assert llm.model.model == "claude-haiku-4-5-20251001"


def test_chat_with_fake_model(mocker, fake_model):
    llm = MovieRecommenderLLM(model=fake_model)
    assert len(llm._conversation) == 3  # Initial messages
    response = llm.chat("Hi, can you help me find a movie to watch tonight?")
    assert response == "What about **The Holy Mountain**?"
    assert len(llm._conversation) == 5  # Initial + user + response
