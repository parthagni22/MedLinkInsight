import pickle
from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Change this to a strong secret key

# Set up MongoDB client
client = MongoClient('mongodb://localhost:27017/')
db = client['registration_db']
patients_collection = db['patients']
admin_collection = db['admin']
advertisements_collection = db['advertisements']

# Define constants
CALORIE_GOAL_LIMIT = 3000
PROTEIN_GOAL = 180
FAT_GOAL = 80
CARBS_GOAL = 300
PAGE_SIZE = 10  # Number of articles per page
NEWS_API_KEY = 'your_news_api_key'

# Add cache control headers to prevent browser caching
@app.after_request
def add_cache_control(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# --------------------USER LOGIN---------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if user exists in the database
        user = patients_collection.find_one({'username': username, 'password': password})

        if user:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html')

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

# --------------------- ADMIN LOGIN ---------------------------------
ADMIN_USERNAME = 'Admin'
ADMIN_PASSWORD = 'Admin@123'

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if 'username' in session:
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if user is an admin
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['username'] = username
            session['password'] = True
            return redirect(url_for('admin_dashboard'))

        # Check if user exists in the database
        user = admin_collection.find_one({'username': username, 'password': password})

        if user:
            session['username'] = username
            session['is_admin'] = False
            return redirect(url_for('index'))

        flash('Invalid username or password', 'error')

    return render_template('login.html')

# ----------------------- ADMIN DASHBOARD --------------------------------
@app.route('/admin_dashboard')
def admin_dashboard():
    if 'username' in session and session.get('is_admin', False):
        # Query MongoDB for pending advertisement requests
        # Display pending requests and provide option for approval
        return render_template('admin_dashboard.html', username=session['username'])
    else:
        flash('Unauthorized access', 'error')
        return redirect(url_for('login'))

@app.route('/approve_advertisement', methods=['POST'])
def approve_advertisement():
    # Handle approval of advertisement images by admin
    # Update MongoDB to mark images as approved
    return redirect(url_for('admin_dashboard'))

# ------------------------ SERVICES ROUTE ---------------------------------
@app.route('/services')
def services_route():
    # Query MongoDB for approved advertisement images
    # Pass the list of approved advertisements to the template
    approved_advertisements = advertisements_collection.find({'status': 'approved'})
    return render_template('services.html', advertisements=approved_advertisements)

# ------------------------ VENDOR REGISTRATION AND LOGIN -------------------
# Add routes for vendor registration and login
@app.route('/vendor_register', methods=['GET', 'POST'])
def vendor_register():
    if request.method == 'POST':
        # Process vendor registration form data
        # Save vendor details to MongoDB
        return redirect(url_for('vendor_login'))
    return render_template('vendor_register.html')

@app.route('/vendor_login', methods=['GET', 'POST'])
def vendor_login():
    if request.method == 'POST':
        # Process vendor login form data
        # Verify credentials and log in vendor
        return redirect(url_for('vendor_dashboard'))
    return render_template('vendor_login.html')

# Add route for vendor dashboard
@app.route('/vendor_dashboard')
def vendor_dashboard():
    # Verify if vendor is logged in
    # Render vendor dashboard with image upload form
    return render_template('vendor_dashboard.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Handle image upload from vendor
    # Save image to MongoDB
    return redirect(url_for('vendor_dashboard'))

# ---------------------------- BLOG ROUTE ----------------------------------
@app.route('/blog')
def blog_route():
    return render_template('blog.html')

# ------------------------ DASHBOARD ---------------------------------------
@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        username = session['username']
        show_chat = request.args.get('show_chat')
        return render_template('dashboard.html', username=username, greeting_message='hi,', show_chat=show_chat)
    else:
        return redirect(url_for('login'))

# ------------------------ CALORIE TRACKER ---------------------------------
class Food:
    def __init__(self, name, calories, proteins, fats, carbs):
        self.name = name
        self.calories = calories
        self.proteins = proteins
        self.fats = fats
        self.carbs = carbs

today = []

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

# -------------------------- SKIN CARE -------------------------------------
# Load and preprocess data
df = pd.read_csv('skindataall.csv', index_col=[0])

# Define TF-IDF vectorizer
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Ingredients'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on content
def recommend_products_by_user_features(skintone, eyecolor, skintype, haircolor):
    ddf = df[(df['Skin_Tone'] == skintone) & (df['Eye_Color'] == eyecolor) & (df['Skin_Type'] == skintype) & (df['Hair_Color'] == haircolor)]
    recommendations = ddf[(ddf['Rating_Stars'].notnull())][['Rating_Stars', 'Product_Url', 'Product']]
    recommendations = recommendations.sort_values('Rating_Stars', ascending=False).head(10)
    return recommendations

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

# ----------------------- VIDEO CALLING -----------------------------------
@app.route('/videocall')
def videoCall():
    return render_template('videocall.html')

# ---------------------- DIABETES PREDICTION -----------------------------
# Load the diabetes prediction model
classifier = joblib.load('svm_diabetes_classifier.sav')
scaler = joblib.load('scaler.sav')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        input_data = request.form.values()
        input_data_as_numpy_array = np.array(input_data).astype(float)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        std_data = scaler.transform(input_data_reshaped)
        prediction = classifier.predict(std_data)
        if prediction[0] == 0:
            result = 'The person is not diabetic'
        else:
            result = 'The person is diabetic'
        return render_template('dibpre.html', result=result)
    return render_template('dibpre.html', result=None)

#----------------------------------------B. Cancer--------------------------------------------------------


# ---------------------- NEWS APP ----------------------------------------
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

# ------------------------- HOME PAGE -------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
