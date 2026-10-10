from typing import Tuple
def split_data(data, test_size) -> Tuple:
    split_at = int(len(data) * (1 - test_size))
    train, test = data.iloc[:split_at], data.iloc[split_at:]
    return train, test
