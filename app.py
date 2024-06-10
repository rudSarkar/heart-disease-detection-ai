from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, EqualTo, ValidationError
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    medical_history = db.Column(db.Text, nullable=True)
    is_admin = db.Column(db.Boolean, default=False)

with app.app_context():
    db.create_all()

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Column transformer and scaler should be defined as used during training
categorical = ['sex', 'cp', 'restecg', 'slope', 'thal']
do_not_touch = ['fbs', 'exang']
non_categorical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

# Ensure consistent categories
dataset = pd.read_csv('heart.csv')
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorical)],
    remainder='passthrough'
)
X = ct.fit_transform(dataset[categorical + do_not_touch + non_categorical])
scaler = StandardScaler()
scaler.fit(X[:, -6:])

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Forms
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired(), EqualTo('confirm', message='Passwords must match')])
    confirm = PasswordField('Confirm Password')
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

class ResetPasswordForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('New Password', validators=[DataRequired(), EqualTo('confirm', message='Passwords must match')])
    confirm = PasswordField('Confirm New Password')
    submit = SubmitField('Reset Password')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('profile'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        user = User(username=form.username.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created!', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html', form=form)

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            hashed_password = generate_password_hash(form.password.data)
            user.password = hashed_password
            db.session.commit()
            flash('Your password has been updated!', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username not found', 'danger')
    return render_template('reset_password.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return render_template('disease_detection.html')
    elif request.method == 'POST':
        data = request.form.to_dict()

        input_data = [data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'], data['fbs'], data['restecg'], data['thalach'], data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']]
        
        input_data = [float(i) for i in input_data]
        
        input_df = pd.DataFrame([input_data], columns=non_categorical + do_not_touch + categorical)
        input_transformed = ct.transform(input_df)
        input_transformed[:, -6:] = scaler.transform(input_transformed[:, -6:])
        
        prediction = model.predict(input_transformed)
        
        result = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"

        current_user.age = data['age']
        current_user.gender = data['sex']
        current_user.medical_history = result
        db.session.commit()
        
        return render_template('result.html', prediction=result)

# Admin Panel
@app.route('/admin', methods=['GET'])
@login_required
def admin_panel():
    if not current_user.is_admin:
        flash('You are not authorized to access the admin panel.', 'danger')
        return redirect(url_for('home'))
    users = User.query.all()
    return render_template('admin_panel.html', users=users)

if __name__ == "__main__":
    app.run(debug=True)
