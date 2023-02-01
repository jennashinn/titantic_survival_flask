from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import (SelectField,  IntegerField, RadioField, SubmitField)
from wtforms.validators import InputRequired
import pandas as pd


import pickle 
rf_model = pickle.load(open('rf_classifier.pkl', 'rb'))


application = Flask(__name__)
application.config['SECRET_KEY'] = 'secret'

@application.before_first_request
def startup():
    global rf_model
    rf_model = pickle.load(open('rf_classifier.pkl', 'rb'))


# model related variables
features = ['Pclass', 'Sex', 'Age', 'SibSp']


class InputForm(FlaskForm):
    sex = RadioField("Please select a sex:", choices=[(0,'Male'),(1,'Female')],
                     validators=[InputRequired()])
    age = IntegerField("What is your age:", 
                       validators=[InputRequired()])
    pclass = SelectField("Which class are you in:", 
                         choices=[(1, 'First'), (2, 'Second'), (3, 'Third')])
    
    sibsp = IntegerField("Number of travelers are in your party:", 
                         validators=[InputRequired()])
   
    submit = SubmitField()


@application.route('/', methods=('GET', 'POST'))
def index():
    probability = 0 
    form = InputForm()
    x_user = []
    if form.validate_on_submit():
        x_user = pd.DataFrame([[
            int(form.pclass.data),
            int(form.sex.data),
            form.age.data,
            form.sibsp.data]], columns = features)
        
        #score = rf_model.predict(x_user[features])
        probs =  rf_model.predict_proba(x_user[features])
        probability = (probs[0,1])*100
        #prob = (probs[0,1])*100

        
    return render_template('index.html',
                           x_user=x_user, 
                           form=form, 
                           prob = probability)


if __name__ == "__main__":
    application.debug = True
    application.run()
