from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask application
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = '2753b167d0de6d94420b18422a417389'

# Initialize SQLAlchemy and Bcrypt for database interaction and password hashing
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Load datasets
testing_df = pd.read_csv('Testing.csv')
training_df = pd.read_csv('Training.csv')
description_df = pd.read_csv('Symptom_description.csv', header=None, names=['Condition', 'Description'])
precaution_df = pd.read_csv('Symptom_precaution.csv', header=None, names=['Condition', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'])
doctors_df = pd.read_csv('doctors_dataset.csv')

# Create dictionaries for disease descriptions and precautions
disease_descriptions = dict(zip(description_df['Condition'], description_df['Description']))
disease_precautions = {row['Condition']: [row['Precaution_1'], row['Precaution_2'], row['Precaution_3'], row['Precaution_4']]
                       for _, row in precaution_df.iterrows()}

# Function to preprocess data for the model
def preprocess_data(df):
    X = df.drop(columns=['prognosis'])
    y = df['prognosis']
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y, le

# Preprocess training data and initialize the classifier
X_train, y_train, label_encoder = preprocess_data(training_df)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Function to predict disease based on symptoms
def predict_disease(symptoms):
    symptom_data = {col: 0 for col in testing_df.columns[:-1]}
    for symptom in symptoms:
        if symptom in symptom_data:
            symptom_data[symptom] = 1
    symptom_df = pd.DataFrame([symptom_data])
    prediction = clf.predict(symptom_df)
    predicted_disease = label_encoder.inverse_transform(prediction)[0]
    return predicted_disease

# User model for the database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(50), nullable=False)
    medical_conditions = db.Column(db.Text, nullable=True)

# Function to create database tables
def create_tables():
    with app.app_context():
        db.create_all()

# Route for the home page
@app.route('/')
def home():
    return redirect(url_for('login'))

# Route for user signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        age = request.form['age']
        gender = request.form['gender']
        medical_conditions = request.form['medical_conditions']
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(
            username=username,
            email=email,
            password=hashed_password,
            age=int(age),
            gender=gender,
            medical_conditions=medical_conditions
        )
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('main'))
        else:
            return render_template('login.html', error='Invalid email or password')
    return render_template('login.html')

# Route for the main user page
@app.route('/main')
def main():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = User.query.get(user_id)
    health_information = {
        'username': user.username,
        'email': user.email,
        'profile': {
            'age': user.age,
            'gender': user.gender,
            'conditions': user.medical_conditions,
        }
    }
    return render_template('main.html', user=user, health_info=health_information)

# Route to check symptoms
@app.route('/check_symptoms', methods=['GET', 'POST'])
def check_symptoms():
    return redirect(url_for('predict'))

# Route to predict disease based on symptoms
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')
        if not selected_symptoms:
            return render_template('predict.html', symptoms=list(training_df.columns[:-1])) 
        
        symptoms = [symptom.strip().replace(' ', '_') for symptom in selected_symptoms]
        predicted_disease = predict_disease(symptoms)
        disease_description = disease_descriptions.get(predicted_disease, "Description not available.")
        precautions = disease_precautions.get(predicted_disease, ["Precautions not available."])

        # Suggest a doctor
        doctor_info = doctors_df.sample(1).iloc[0] # Selects one random row from doctors_df
        doctor_name = doctor_info[0]
        doctor_url = doctor_info[1]

        return render_template('predict.html', prediction=predicted_disease, description=disease_description, precautions=precautions, doctor_name=doctor_name, doctor_url=doctor_url, symptoms=list(training_df.columns[:-1]))
    
    return render_template('predict.html', symptoms=list(training_df.columns[:-1]))

# Route to get symptoms for autocomplete feature
@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    query = request.args.get('query', '').lower()
    all_symptoms = list(training_df.columns[:-1])
    filtered_symptoms = [symptom for symptom in all_symptoms if query in symptom.lower()]
    return jsonify({'symptoms': filtered_symptoms})

# Route to display common diseases
@app.route('/common-diseases')
def common_diseases():
    return render_template('common-diseases.html')

# Route to display emergency contacts
@app.route('/emergency-contacts')
def emergency_contacts():   
    return render_template('emergency-contacts.html')

# Route to display first aid tips
@app.route('/first-aid-tips')
def first_aid_tips():
    return render_template('first-aid-tips.html')

# Main function to create tables and run the application
if __name__ == "__main__":
    create_tables()
    app.run(debug=True)
