import pandas as pd

titanic_df = pd.read_csv('train.csv')


titanic_df.rename(columns={'PassengerId': 'passenger_id',
                   'Survived': 'survived',
                   'Pclass': 'pclass',
                   'Name': 'name',
                   'Sex': 'sex',
                   'Age': 'age',
                   'SibSp': 'sibsp',
                   'Parch': 'parch',
                   'Ticket': 'ticket',
                   'Fare': 'fare',
                   'Cabin': 'cabin',
                   'Embarked': 'embarked'}, inplace=True)

titanic_df.embarked.fillna('', inplace=True)
titanic_df.cabin.fillna('', inplace=True)
titanic_df['cabin'] = titanic_df['cabin'].astype(str).str[0]
titanic_df.cabin.fillna('', inplace=True)
titanic_df.fillna(0, inplace=True)
titanic_df.drop(['name', 'passenger_id', 'ticket'], axis=1, inplace=True)