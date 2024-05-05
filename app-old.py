import pickle
from flask import Flask, render_template, request, redirect, url_for, make_response, session, jsonify, flash
import joblib
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


from flaskext.mysql import MySQL
import requests

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Change this to a strong secret key
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'Ashirvad@123'
app.config['MYSQL_DATABASE_DB'] = 'medlinkinsight'

client = MongoClient('mongodb://localhost:27017/')
db = client['registration_db']
doctors_collection = db['doctors']
patients_collection = db['patients']
admin_collection = db['admin']
advertisements_collection = db['Adv']

mysql = MySQL(app)

# -------------------  Registration --------------------------

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')

        # Check if the username already exists in the database
        if patients_collection.find_one({'username': username}):
            return 'Username already exists. Choose a different one.'
        data = {
            'full_name': request.form.get('full_name'),
            'username': request.form.get('username'),
            'dob': request.form.get('dob'),
            'marital_status': request.form.get('marital_status'),
            'address': request.form.get('address'),
            'phone_no': request.form.get('phone_no'),
            'email': request.form.get('email'),
            'allergies': request.form.get('allergies'),
            'medication': request.form.get('medication'),
            'password': request.form.get('password'),
        }

        # Insert patient data into MongoDB
        patients_collection.insert_one(data)

        return 'Data updated to MongoDB'

    return render_template('register.html')

# --------------------USER LOGIN Admin Login---------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_type = request.form['user_type']

        # Check if the entered credentials belong to a patient
        if user_type == 'patient':
            user = patients_collection.find_one({'username': username, 'password': password})
            if user:
                # If the credentials belong to a patient, set up the session and redirect to patient dashboard
                session['logged_in'] = True
                session['user_type'] = 'patient'
                session['username'] = username
                return redirect(url_for('dashboard'))

        # Check if the entered credentials belong to an admin
        elif user_type == 'admin':
            user = admin_collection.find_one({'username': username, 'password': password})
            if user:
                # If the credentials belong to an admin, set up the session and redirect to admin dashboard
                session['logged_in'] = True
                session['user_type'] = 'admin'
                session['username'] = username
                return redirect(url_for('admin_dashboard'))

        # If credentials are incorrect or do not belong to either a patient or an admin, show an error message
        flash('Invalid username or password', 'error')

    # Render the login page template
    return render_template('login.html')


# Dashboard route
@app.route('/dashboard')
def dashboard():
    # Check if a user is logged in
    if 'logged_in' not in session or not session['logged_in']:
        # If not logged in, redirect to the login page
        return redirect(url_for('login'))

    # Check if the user is a patient
    if session['user_type'] == 'patient':
        greeting_message = "Welcome to Dashboard"
        username = "User"
        # Render the patient dashboard template
        return render_template('dashboard.html',greeting_message=greeting_message, username=username)

    # # If the user is an admin, redirect to the admin dashboard
    # return redirect(url_for('admin_dashboard'))

# Admin dashboard route
@app.route('/admin_dashboard')
def admin_dashboard():
    # Check if a user is logged in
    if 'logged_in' not in session or not session['logged_in']:
        # If not logged in, redirect to the login page
        return redirect(url_for('login'))

    # Check if the user is an admin
    if session['user_type'] == 'admin':
        # Render the admin dashboard template
        return render_template('admin_dashboard.html')

    # # If the user is not an admin, redirect to the patient dashboard
    # return redirect(url_for('dashboard'))


@app.route('/logout')
def logout():
    # Clear session
    session.clear()
    flash("You have been logged out", "Success")
    return redirect(url_for('index'))

# -------------------- blog Page---------------------------------


@app.route('/blog')
def blog_rout():
    # Fetch images
    images = advertisements_collection.find()
    return render_template('blog.html', images=images)

#-----------------------------  Advt.----------------------------------------

# Route for uploading advertisement image
@app.route('/upload_advertisement', methods=['POST'])
def upload_advertisement():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'advertisement_image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['advertisement_image']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # If the file exists and is allowed, save it to the database
        if file:
            advertisements_collection.insert_one({'advertisement_image': file.read()})
            flash('File uploaded successfully')
            return redirect(url_for('admin_dashboard'))




# ------------------------------ Calorie Tracker --------------------------------
CALORIE_GOAL_LIMIT = 3000
PROTEIN_GOAL = 180
FAT_GOAL = 80
CARBS_GOAL = 300

today = []

class Food:
    def __init__(self, name, calories, proteins, fats, carbs):
        self.name = name
        self.calories = calories
        self.proteins = proteins
        self.fats = fats
        self.carbs = carbs

@app.route('/calorie')
def calorie():
    return render_template('calorie.html')

@app.route('/add_food', methods=['POST'])
def add_food():
    name = request.form['name']
    calories = int(request.form['calories'])
    proteins = int(request.form['proteins'])
    fats = int(request.form['fats'])
    carbs = int(request.form['carbs'])

    food = Food(name, calories, proteins, fats, carbs)
    today.append(food)

    return render_template('calorie.html', today=today)

@app.route('/visualize')
def visualize():
    calorie_sum = sum(food.calories for food in today)
    protein_sum = sum(food.proteins for food in today)
    fats_sum = sum(food.fats for food in today)
    carbs_sum = sum(food.carbs for food in today)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].pie([protein_sum, fats_sum, carbs_sum], labels=["Proteins", "Fats", "Carbs"], autopct="%1.1f%%")
    axs[0, 0].set_title("Macronutrients Distribution")
    axs[0, 1].bar([0, 1, 2], [protein_sum, fats_sum, carbs_sum], width=0.4)
    axs[0, 1].bar([0.5, 1.5, 2.5], [PROTEIN_GOAL, FAT_GOAL, CARBS_GOAL], width=0.4)
    axs[0, 1].set_title("Macronutrients Progress")
    axs[1, 0].pie([calorie_sum, CALORIE_GOAL_LIMIT - calorie_sum], labels=["Calories", "Remaining"], autopct="%1.1f%%")
    axs[1, 0].set_title("Calories Goal Progress")
    axs[1, 1].plot(list(range(len(today))), np.cumsum([food.calories for food in today]), label="Calories Eaten")
    axs[1, 1].plot(list(range(len(today))), [CALORIE_GOAL_LIMIT] * len(today), label="Calorie Goal")
    axs[1, 1].legend()
    axs[1, 1].set_title("Calories Goal Over Time")
    fig.tight_layout()

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    # Convert the plot to base64 for embedding in HTML
    plot_data = base64.b64encode(img.getvalue()).decode()

    return render_template('visualize.html', plot_data=plot_data)

# ------------------ Skin Care--------------------------------

# Load and preprocess data
df = pd.read_csv('skindataall.csv', index_col=[0])

# Define TF-IDF vectorizer
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Ingredients'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on content
def recommend_products_by_user_features(skintone, eyecolor, skintype, haircolor):
    ddf = df[(df['Skin_Tone'] == skintone) & (df['Eye_Color'] == eyecolor) & (df['Skin_Type'] == skintype) & (df['Hair_Color'] == haircolor)]
    recommendations = ddf[(ddf['Rating_Stars'].notnull())][['Rating_Stars', 'Product_Url', 'Product']]
    recommendations = recommendations.sort_values('Rating_Stars', ascending=False).head(10)
    return recommendations

# Function to get recommendations based on content
def content_recommendation(product):
    idx = df[df['Product'] == product].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices]

@app.route('/skincare')
def skincare():
    return render_template('skincare.html')

@app.route('/recommendation', methods=['POST'])
def recommendations():
    skintone = request.form['skintone']
    eyecolor = request.form['eyecolor']
    skintype = request.form['skintype']
    haircolor = request.form['haircolor']
    recommendations = recommend_products_by_user_features(skintone, eyecolor, skintype, haircolor)
    return render_template('recommendation.html', recommendations=recommendations)



# ---------------------------- Video Calling -----------------------------------------------
@app.route('/videocall')
def videoCall():
    return render_template('videocall.html')



#------------------------------- Dib. prediction---------------------------
with open("Dib_model.pkl","rb") as f:
    model = pickle.load(f)
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user input from the form
        Pregnancies = int(request.form["Pregnancies"])
        Glucose = int(request.form["Glucose"])
        BloodPressure = int(request.form["BloodPressure"])
        SkinThickness = int(request.form["SkinThickness"])
        Insulin = int(request.form["Insulin"])
        BMI = float(request.form["BMI"])
        DiabetesPedigreeFunction = float(request.form["DiabetesPedigreeFunction"])
        Age = int(request.form["Age"])

        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            "Pregnancies": [Pregnancies],
            "Glucose": [Glucose],
            "BloodPressure": [BloodPressure],
            "SkinThickness": [SkinThickness],
            "Insulin": [Insulin],
            "BMI": [BMI],
            "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
            "Age": [Age]
        })

        # Get the prediction
        user_result = model.predict(user_data)

        # Determine the output message
        if user_result[0] == 0:
            output = "Great Non - Dibetic!!"
        else:
            output = "You are Dibetic"

        # Render the result page with the output message
    return render_template('dibpredict.html', output=output)

@app.route('/dibpredict.html')
def dib_prediction():
    return render_template('dibpredict.html')


#-------------------------------- B. Cancer----------------
with open("B_cancer_model.pkl","rb") as f:
    bmodel = pickle.load(f)
@app.route("/b_cancer_model", methods=["POST"])
def b_cancer_model():
    if request.method =="POST":
        # Get user input from the form
        mean_radius = float(request.form['mean_radius'])
        mean_texture = float(request.form['mean_texture'])
        mean_perimeter = float(request.form['mean_perimeter'])
        mean_area = float(request.form['mean_area'])
        mean_smoothness = float(request.form['mean_smoothness'])
        mean_compactness = float(request.form['mean_compactness'])
        mean_concavity = float(request.form['mean_concavity'])
        mean_concave_points = float(request.form['mean_concave_points'])
        mean_symmetry = float(request.form['mean_symmetry'])
        mean_fractal_dimension = float(request.form['mean_fractal_dimension'])
        radius_error = float(request.form['radius_error'])
        texture_error = float(request.form['texture_error'])
        perimeter_error = float(request.form['perimeter_error'])
        area_error = float(request.form['area_error'])
        smoothness_error = float(request.form['smoothness_error'])
        compactness_error = float(request.form['compactness_error'])
        concavity_error = float(request.form['concavity_error'])
        concave_points_error = float(request.form['concave_points_error'])
        symmetry_error = float(request.form['symmetry_error'])
        fractal_dimension_error = float(request.form['fractal_dimension_error'])
        worst_radius = float(request.form['worst_radius'])
        worst_texture = float(request.form['worst_texture'])
        worst_perimeter = float(request.form['worst_perimeter'])
        worst_area = float(request.form['worst_area'])
        worst_smoothness = float(request.form['worst_smoothness'])
        worst_compactness = float(request.form['worst_compactness'])
        worst_concavity = float(request.form['worst_concavity'])
        worst_concave_points = float(request.form['worst_concave_points'])
        worst_symmetry = float(request.form['worst_symmetry'])
        worst_fractal_dimension = float(request.form['worst_fractal_dimension'])

        user_data = pd.DataFrame({
            'mean_radius': [mean_radius],
            'mean_texture': [mean_texture],
            'mean_perimeter': [mean_perimeter],
            'mean_area': [mean_area],
            'mean_smoothness': [mean_smoothness],
            'mean_compactness': [mean_compactness],
            'mean_concavity': [mean_concavity],
            'mean_concave_points': [mean_concave_points],
            'mean_symmetry': [mean_symmetry],
            'mean_fractal_dimension': [mean_fractal_dimension],
            'radius_error': [radius_error],
            'texture_error': [texture_error],
            'perimeter_error': [perimeter_error],
            'area_error': [area_error],
            'smoothness_error': [smoothness_error],
            'compactness_error': [compactness_error],
            'concavity_error': [concavity_error],
            'concave_points_error': [concave_points_error],
            'symmetry_error': [symmetry_error],
            'fractal_dimension_error': [fractal_dimension_error],
            'worst_radius': [worst_radius],
            'worst_texture': [worst_texture],
            'worst_perimeter': [worst_perimeter],
            'worst_area': [worst_area],
            'worst_smoothness': [worst_smoothness],
            'worst_compactness': [worst_compactness],
            'worst_concavity': [worst_concavity],
            'worst_concave_points': [worst_concave_points],
            'worst_symmetry': [worst_symmetry],
            'worst_fractal_dimension': [worst_fractal_dimension]

        })

        # Get the prediction
        user_result = bmodel.predict(user_data)
        if user_result[0] ==0:
            output = "Malignantic"
        else:
            output = "Benign"
    return render_template('b_cancer_model.html', output=output)

@app.route("/b_cancer_model.html")
def b_cancer_prediction():
    return render_template('b_cancer_model.html')

#---------------------------------- C. Heart Disease---------------------------
with open("hrt_model.pkl","rb") as f:
    hmodel = pickle.load(f)
@app.route("/heart_model", methods=["POST"])
def heart_model():
    if request.method =="POST":

        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        user_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction= hmodel.predict(user_data)

        if prediction[0] == 0:
            output = "You are Healthy"
        else:
            output = "Heart Disease Ditected"
    return render_template('hrt_disease.html', output=output)

@app.route('/hrt_disease.html')
def hrt_prediction():
    return render_template('hrt_disease.html')







# --------------------------------------------- News App---------------------------------------------

NEWS_API_KEY= '6280f9db4c5245878a84229a1ee84f11'
PAGE_SIZE = 10  # Number of articles per page

@app.route('/newsapp')
def newsapp():
    page = int(request.args.get('page', 1))  # Get the page number from query parameters
    start_index = (page - 1) * PAGE_SIZE
    end_index = start_index + PAGE_SIZE

    # Fetch top Headlines
    url = f'https://newsapi.org/v2/top-headlines?language=en&category=health&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    data = response.json()
    articles = data['articles']

    # Paginate articles
    paginated_articles = articles[start_index:end_index]

    num_articles = len(articles)
    num_pages = num_articles // PAGE_SIZE + (1 if num_articles % PAGE_SIZE > 0 else 0)

    return render_template('newsapp.html', articles=paginated_articles, num_articles=num_articles, num_pages=num_pages, current_page=page)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/services')
def services():
    return render_template('services.html')



if __name__ == '__main__':
    app.run(debug=True)
