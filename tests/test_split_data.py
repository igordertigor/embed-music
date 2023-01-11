import pandas as pd

from src.split_data import split_threefold


def test_split_threefold():
    data = pd.DataFrame(
        {'items': [1]*100, 'genre_top': [1, 2, 3, 4]*25},
    )

    train, valid, test = split_threefold(data, .1, .1)

    assert len(train) == 80
    assert len(valid) == 10
    assert len(test) == 10
