import pandas as pd
from random import choice

if __name__ == '__main__':
    df = pd.read_csv('building_20230808.csv')
    df['is_updated'] = [choice([True, False]) for _ in range(len(df))]
    df.to_csv('building_20230808.csv', index=False)