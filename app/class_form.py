from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField, IntegerField, FloatField
from wtforms.validators import DataRequired, NumberRange

class ModelForm(FlaskForm):

    name = StringField('Name', validators=[DataRequired()])
    pclass = SelectField('Passenger class', choices=[('1', '1st class'), ('2', '2nd class'), ('3', '3rd class')], validators=[DataRequired()])
    age = IntegerField('Age', [NumberRange(min=0, max=80)])
    sex = SelectField('Gender', choices=[('male','M'), ('female','F')], validators=[DataRequired()])
    sibsp = IntegerField('Number of siblings or spouses on board', [NumberRange(min=0, max=8)])
    parch = IntegerField('Number of parents or children on board', [NumberRange(min=0, max=6)])
    fare = FloatField('Ticket fare', [NumberRange(min=0, max=550)])
    cabin = SelectField('Cabin deck', choices=[('A', 'A'), ('B', 'B'), ('C', 'C'), ('D', 'D'), ('E', 'E'), ('F', 'F'), ('G', 'G'), ('T', 'T')], validators=[DataRequired()])
    embarked = SelectField('Embarked in', choices=[('C','Cherbourg'), ('Q','Queenstown'), ('S','Southampton')], validators=[DataRequired()])
    submit = SubmitField('Did I survive???')





