

import pandas as pd
from joblib import dump, load
from prediction_class import PredictionModel
from insert_parameters import get_model_params


model = PredictionModel(
    selected_models='rf,lgbm,mlp',
    model_grid_list=get_model_params(),
    search_method='random',
    search_iters=1,
    search_scorer='f1',
    ensemble_method='soft_voting',
    verbose=3)

model.fit_full()
model.results
model.predict_kaggle()

#dump(model.model_search_result, "best_titanic_predictor.pkl")
#model = load("best_titanic_predictor.pkl")
dump(model.model_search_result, open('best_titanic_predictor.pkl', 'wb'))

