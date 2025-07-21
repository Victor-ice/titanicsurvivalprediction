import pandas as pd

def add_title(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
    # Group rare titles into 'Other'
    rare_titles = df['Title'].value_counts()[lambda x: x < 10].index
    df['Title'] = df['Title'].replace(rare_titles, 'Other')
    return df

def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df

def add_deck(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Deck'] = df['Cabin'].str[0].fillna('U')  # U = Unknown
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = add_title(df)
    df = add_family_features(df)
    df = add_deck(df)

    # Fill missing Embarked
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Impute missing Age and Fare
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # One‑hot encode categorical vars
    cat_cols = ['Pclass','Sex','Embarked','Title','Deck']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

if __name__ == "__main__":
    from src.data_loader import load_titanic_data
    train_df, _ = load_titanic_data()
    processed = preprocess(train_df)
    print("Processed shape:", processed.shape)
    print(processed.columns)
