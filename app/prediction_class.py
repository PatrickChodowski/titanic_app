
class PredictionModel:

    def __init__(self,
                selected_models,
                model_grid_list,
                search_method,
                search_iters,
                search_scorer,
                ensemble_method,
                verbose):

        self.selected_models = selected_models
        self.ensemble_method = ensemble_method
        self.search_method = search_method
        self.model_grid_list = model_grid_list
        self.search_iters = search_iters
        self.search_scorer = search_scorer
        self.njobs = 6
        self.verbose = verbose
        self.df = None
        self.pipeline = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model_params = None
        self.column_transformer = None
        self.selected_models_list = None
        self.search_object = None
        self.estimator = None
        self.model_search_result = None
        self.predictions_table = None
        self.results = None

    def build_datasets(self):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        df = pd.read_csv("train.csv")

        df.rename(columns={'PassengerId':'passenger_id',
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
                           'Embarked':'embarked'},inplace=True)

        df.embarked.fillna('', inplace=True)
        df.cabin.fillna('', inplace=True)
        df['cabin'] = df['cabin'].astype(str).str[0]
        df.cabin.fillna('', inplace=True)

        df.fillna(0, inplace=True)
        self.df = df[[c for c in list(df) if len(df[c].unique()) > 1]]
        x = self.df.drop(['survived','name','passenger_id','ticket'], axis=1)
        y = self.df['survived']

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=0)


    def build_column_transformer(self):

        from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder, KBinsDiscretizer
        from sklearn.compose import make_column_transformer

        self.column_transformer = make_column_transformer(
            (KBinsDiscretizer(), ['sibsp', 'parch']),
            (RobustScaler(), ['fare', 'age']),
            (OneHotEncoder(handle_unknown='ignore'), ['pclass', 'sex', 'embarked', 'cabin']),
            n_jobs=-self.njobs,
            remainder='passthrough'
        )

    def build_estimator(self):

        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import VotingClassifier
        from mlxtend.classifier import StackingClassifier

        self.selected_models_list = self.selected_models.split(',')
        estimator_list = []
        full_model_params = []

        for idx, estimator in enumerate(self.selected_models_list):

            if estimator == 'rf':
                estimator_object = RandomForestClassifier(n_jobs=self.njobs, verbose=self.verbose)
            elif estimator == 'glm':
                estimator_object = LogisticRegression(n_jobs=self.njobs, verbose=self.verbose)
            elif estimator == 'lgbm':
                estimator_object = LGBMClassifier(n_jobs=self.njobs, verbose=3)
            elif estimator == 'mlp':
                estimator_object = MLPClassifier(verbose=self.verbose)

            model_params = dict(
                [(key, value) for key, value in self.model_grid_list.items() if key.startswith(estimator)])

            if self.ensemble_method == 'stacking':
                if estimator == 'rf':
                    estimator_name = 'randomforest'
                else:
                    estimator_name = estimator
                model_param_name = 'estimator__' + estimator_name + 'classifier'
                for k, v in list(model_params.items()):
                    model_params[k.replace(estimator, model_param_name)] = model_params.pop(k)
                estimator_list.append(estimator_object)
                full_model_params.append(model_params)

            elif self.ensemble_method in ['soft_voting', 'hard_voting']:

                model_param_name = 'estimator__' + estimator
                for k, v in list(model_params.items()):
                    model_params[k.replace(estimator, model_param_name)] = model_params.pop(k)

                full_model_params.append(model_params)
                tpp = estimator, estimator_object
                estimator_list.append(tpp)

        self.model_params = {}
        for d in full_model_params:
            self.model_params.update(d)

        if self.ensemble_method == 'soft_voting':
            self.estimator = VotingClassifier(estimator_list, voting='soft', n_jobs=self.njobs)

        elif self.ensemble_method == 'hard_voting':
            self.estimator = VotingClassifier(estimator_list, voting='hard', n_jobs=self.njobs)

        elif self.ensemble_method == 'stacking':

            lr = LogisticRegression()
            self.estimator = StackingClassifier(estimator_list, meta_classifier=lr)

        elif self.ensemble_method is None:
            self.estimator = estimator_list[0]


    def build_pipeline(self):
        from sklearn.pipeline import Pipeline

        self.pipeline = Pipeline(
            steps=[
                ('scaling', self.column_transformer),
                ('estimator', self.estimator)
            ], verbose=True)

    def build_search(self, cv_=5):
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        if self.search_method == 'random':
            self.search_object = RandomizedSearchCV(
                estimator=self.pipeline,
                n_iter=self.search_iters,
                param_distributions=self.model_params,
                refit=True,
                cv=cv_,
                scoring=self.search_scorer,
                error_score=-1,
                verbose=3
            )

        elif self.search_method == 'grid':
            self.search_object = GridSearchCV(
                estimator=self.pipeline,
                param_grid=self.model_params,
                refit=True,
                cv=cv_,
                scoring=self.search_scorer,
                error_score=-1,
                verbose=3)

    def fit(self):
            self.model_search_result = self.search_object.fit(self.x_train, self.y_train.values.ravel())

    def predict(self):

        import pandas as pd
        import datetime as dttt

        y_test_df = pd.DataFrame(self.y_test).reset_index()
        y_test_df['prediction_date'] = str(dttt.date.today())
        y_test_df['pred_05'] = self.model_search_result.predict(self.x_test)
        y_test_df['pred_proba'] = self.model_search_result.predict_proba(self.x_test)[:, 1].round(4)


        self.predictions_table = y_test_df



    def predict_kaggle(self):
        import pandas as pd
        df = pd.read_csv("test.csv")

        df.rename(columns={'PassengerId': 'passenger_id',
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

        df.embarked.fillna('', inplace=True)
        df.cabin.fillna('', inplace=True)
        df['cabin'] = df['cabin'].astype(str).str[0]
        df.cabin.fillna('', inplace=True)
        df.fillna(0, inplace=True)
        self.df = df[[c for c in list(df) if len(df[c].unique()) > 1]]

        kaggle_test_dfx = self.df.drop(['name', 'passenger_id', 'ticket'], axis=1)

        kaggle_pred = self.model_search_result.predict(kaggle_test_dfx)

        submission = pd.DataFrame({
            "PassengerId": df["passenger_id"],
            "Survived": kaggle_pred
        })
        submission.to_csv('titanic_kaggle_results.csv', index=False)



    def calculate_results(self):
        from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

        self.results = dict()
        self.results['val_acc'] = accuracy_score(self.predictions_table.survived,
                                                                 self.predictions_table.pred_05)
        self.results['val_rec'] = recall_score(self.predictions_table.survived,
                                                               self.predictions_table.pred_05)
        self.results['val_pres'] = precision_score(self.predictions_table.survived,
                                                                   self.predictions_table.pred_05)
        self.results['val_f1'] = f1_score(self.predictions_table.survived, self.predictions_table.pred_05)



    def fit_full(self):

        self.build_datasets()
        self.build_column_transformer()
        self.build_estimator()
        self.build_pipeline()
        self.build_search()

        self.fit()
        self.predict()
        self.calculate_results()

        #self.run_lime()

