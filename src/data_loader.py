import pandas as pd
from pathlib import Path

def load_titanic_data(data_dir: str = "V:/titanic") -> pd.DataFrame:
    """
    Loads csv titanic data
    Expects 'train.csv' and 'test.csv'
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path  = data_dir / "test.csv"

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    return df_train, df_test

if __name__ == "__main__":
    train, test = load_titanic_data()
    print("Train shape:", train.shape)
    print("Test shape: ", test.shape)
