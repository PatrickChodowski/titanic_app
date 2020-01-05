import json
import requests

url = 'http://127.0.0.1:5000/predict_survive'

data = {'pclass': 3,
        'sex': 'male',
        'age': 2,
        'sibsp': 1,
        'parch': 1,
        'fare': 50,
        'cabin': 'A',
        'embarked': 'S'}
data = json.dumps(data)
requests.post(url, data)

data = {'pclass': 3,
        'sex': 'male',
        'age': 2,
        'sibsp': 1,
        'parch': 1,
        'fare': 50,
        'cabin': 'A',
        'embarked': 'S'}
