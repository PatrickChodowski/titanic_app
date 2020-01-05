import os

os.chdir('./app')

from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from joblib import load
import pandas as pd
from app.class_form import ModelForm
from sklearn.preprocessing import _data

import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from app.explain_module import titanic_df
import dash_dangerously_set_inner_html

titanic_app = Flask(__name__)
SECRET_KEY = os.urandom(32)
titanic_app.config['SECRET_KEY'] = SECRET_KEY
titanic_app.config["FLASK_DEBUG"] = 1
model = load(open('best_titanic_predictor.pkl', 'rb'))


def pred_row(request_form):
    pdata = {'pclass': int(request_form.getlist('pclass')[0]),
            'sex': request_form.getlist('sex')[0],
            'age': request_form.getlist('age')[0],
            'sibsp': request_form.getlist('sibsp')[0],
            'parch': request_form.getlist('parch')[0],
            'fare': request_form.getlist('fare')[0],
            'cabin': request_form.getlist('cabin')[0],
            'embarked': request_form.getlist('embarked')[0]}
    return pdata


@titanic_app.route('/')
@titanic_app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@titanic_app.route('/predict_survive', methods=['POST'])
def predict_survive():
    r_data = request.form
    data = pred_row(r_data)
    user_name = r_data.getlist('name')[0]
    data_df = pd.DataFrame(data, index=[0])
    result = int(model.predict(data_df)[0])
    #result = 1
    if result == 1:
        msg = f"{user_name} survived Titanic crash"
    else:
        msg = f"{user_name} did not survive Titanic crash"

    flash(msg)
    return redirect(url_for("predict"))

@titanic_app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = ModelForm()
    return render_template('predict.html', form=form)
    


#### DASH APP #####


dash_app = dash.Dash(__name__, server=titanic_app, routes_pathname_prefix='/explain/')
col_options = [dict(label=x, value=x) for x in titanic_df.columns]
dimensions = ["x", "y", "color", "facet_col", "facet_row"]

dash_app.layout = html.Div(
    [
        #html.H1("Titanic express analysis"),
            dash_dangerously_set_inner_html.DangerouslySetInnerHTML('''

        <header>
            <nav>
                <ul>
                    <li class="logotyp"><a href="/">Home</a></li>
                    <!--li><a href="/#about_me">About me</a></li-->
                    <!--li><a href="/#about_project">About project</a></li-->
                </ul>
         </nav>
        </header>

'''),
        html.Div(
            [
                html.P([d + ":", dcc.Dropdown(id=d, options=col_options)])
                for d in dimensions
            ],
            style={"width": "15%", "float": "left", "margin-right": "20px", "margin-left": "20px", "color": "black"},
        ),
        dcc.Graph(id="graph", style={"width": "60%", "display": "inline-block", "textAlign": "center"}),
    ]
)

@dash_app.callback(Output("graph", "figure"), [Input(d, "value") for d in dimensions])
def make_figure(x, y, color, facet_col, facet_row):
    return px.scatter(
        titanic_df,
        x=x,
        y=y,
        color=color,
        facet_col=facet_col,
        facet_row=facet_row,
        height=650,
    )


#### RUN #####


#if __name__ == '__main__':
#    app.run()

