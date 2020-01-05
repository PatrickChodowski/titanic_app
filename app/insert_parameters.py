
def get_model_params():

    full_param_grid_list = {
        'rf__max_depth': [5,10,15],
        'rf__criterion': ['gini'],
        'rf__n_estimators': [300],
        'rf__min_samples_leaf': [1,5,10,20,60],

        'dtr__max_depth': [5, 10, 25, 50],
        'dtr__criterion': ['gini'],
        'dtr__min_samples_leaf': [1, 100, 200],

        'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.4],
        'xgb__max_depth': [3, 6, 10, 25, 50],
        'xgb__n_estimators': [500],
        'xgb__colsample_bytree': [0.3, 0.5, 0.9],

        'glm__max_iter': [100, 500, 300],

        'lgbm__n_estimators': [300,500,1000],
        'lgbm__learning_rate': [0.01, 0.001],
        'lgbm__min_child_samples': [5,20,30],
        'lgbm__max_depth': [5,10,12],

        'mlp__alpha': [0.1],
        'mlp__activation': ['relu'],
        'mlp__learning_rate': ['constant'],
        'mlp__batch_size': [100, 200, 300, 400],
        'mlp__max_iter': [500],
        'mlp__tol': [0.0001],
        'mlp__hidden_layer_sizes': [(10), (2)]

    }
    return full_param_grid_list