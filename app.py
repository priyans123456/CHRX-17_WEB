import base64
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory
from flask_pymongo import PyMongo
from bson.objectid import ObjectId  # Import ObjectId
import bcrypt
from datetime import datetime
import random
import string
from flask import render_template, request, redirect, url_for, session
import pymongo
from datetime import datetime
import os
from werkzeug.utils import secure_filename
from bson import ObjectId
from functools import wraps
import razorpay
import smtplib
from email.mime.text import MIMEText
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
from flask import Flask, render_template, request, redirect, url_for, session
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

from dotenv import load_dotenv
import os

load_dotenv()





app = Flask(__name__)

# MongoDB Atlas Connection URI
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)


# Check DB connection
try:
    # Try to list database names (will raise error if not connected)
    db_names = mongo.cx.list_database_names()
    print("✅ MongoDB connected! Available databases:", db_names)
except Exception as e:
    print("❌ MongoDB connection failed:", str(e))



# MongoDB Collections
users = mongo.db.users
gradcam=mongo.db.gradcam

app.secret_key = '56f8c7e3b9e4a8d5f12a6c89db307a4f7b3c91c6745d2198'
app.config['SESSION_COOKIE_SECURE'] = True  # Use HTTPS for session cookies
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to session cookies




@app.route('/')
def home():

    return render_template('index.html')



@app.route('/login1')
def login1():
    return render_template('login.html')


# Email Configuration
EMAIL_ADDRESS = "plutonium877@gmail.com"
EMAIL_PASSWORD = "ayew gkqt rqoi yorb"


# Function to generate a 6-digit OTP
def generate_otp():
    return str(random.randint(100000, 999999))


# Function to send OTP via Email
def send_otp(email, otp):
    subject = "Your OTP for Registration"
    body = f"Your OTP for registration is: {otp}. It is valid for 5 minutes."

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print("Failed to send email:", e)
        return False


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            data = request.get_json()  # 🔄 Instead of request.form

            fullname = data.get('fullname')
            phone = data.get('phone')
            email = data.get('email')
            password = data.get('password')
            re_password = data.get('re_pass')

            # Validate required fields
            if not all([fullname, phone, email, password, re_password]):
                return jsonify({'status': 'error', 'message': 'All fields are required'})
            # Check if user exists
            if mongo.db.users.find_one({'$or': [{'phone': phone}, {'email': email}]}):
                return jsonify({'status': 'error', 'message': 'User already exists!'})

            if password != re_password:
                return jsonify({'status': 'error', 'message': 'Passwords do not match'})

            # Generate and send OTP
            otp = generate_otp()
            session['otp'] = otp
            session['registration_data'] = {
                'fullname': fullname,
                'phone': phone,
                'email': email,
                'password': password  # 🔒 Note: hash before storing in real apps
            }

            if send_otp(email, otp):
                return jsonify({'status': 'success', 'message': 'OTP sent to your email'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to send OTP'})

        except Exception as e:
            print(f"Registration error: {str(e)}")
            return jsonify({'status': 'error', 'message': 'An error occurred during registration'})

    return render_template('register.html')


@app.route("/otp_verification")
def otp_verification():
    if 'otp' not in session or 'registration_data' not in session:
        # 👇 Redirect back to register if session data is missing
        return redirect("/register")
    return render_template("otp_verification.html")


@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    user_otp = data.get('otp')

    if user_otp == session.get('otp'):

        registration_data = session.pop('registration_data', None)
        if registration_data:
            # Optional: check if user already exists
            existing_user = mongo.db.users.find_one({'email': registration_data['email']})
            if existing_user:
                return jsonify({'status': 'error', 'redirect_url': '/register', 'message': 'Email already registered'})

            # Hash the password
            hashed_password = generate_password_hash(registration_data['password'])

            # Prepare user document
            user_doc = {
                'fullname': registration_data['fullname'],
                'email': registration_data['email'],
                'phone': registration_data['phone'],
                'password': hashed_password,
                'created_at': datetime.utcnow()
            }

            # ✅ Insert into MongoDB
            mongo.db.users.insert_one(user_doc)

            # Clear session OTP
            session.pop('otp', None)
            session.pop('registration_data', None)

            # ✅ Auto-login: Set session
            session['phone'] = registration_data['phone']

            return jsonify(
                {'status': 'success', 'redirect_url': '/index2', 'message': 'OTP verified and user registered'})
        else:
            return jsonify(
                {'status': 'error', 'redirect_url': '/register', 'message': 'Session expired. Please register again.'})
    else:
        return jsonify({'status': 'error', 'redirect_url': '/register', 'message': 'Invalid OTP'})


@app.route('/resend_otp', methods=['POST'])
def resend_otp():
    try:
        registration_data = session.get('registration_data')
        if not registration_data:
            return jsonify({'status': 'error', 'message': 'Session expired. Please register again.'})

        new_otp = generate_otp()
        session['otp'] = new_otp

        if send_otp(registration_data['email'], new_otp):
            return jsonify({'status': 'success', 'message': 'OTP resent successfully.'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to send OTP.'})
    except Exception as e:
        print(f"Resend OTP error: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Server error'})


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        phone = request.form['phone']
        password = request.form['password']

        # Fetch user from MongoDB
        user = users.find_one({'phone': phone})

        # Compare entered password with stored hashed password
        if user and check_password_hash(user['password'], password):
            session['phone'] = phone  # Store phone number in session

            return redirect(url_for('index2'))
        else:
            return 'Invalid credentials! *PLEASE SIGN UP*'

    return render_template('login.html')  # Render the login page for GET request


# --------------------------------
# ✅ Logout Route
@app.route('/logout')
def logout():
    session.clear()  # Clear session data
    return redirect(url_for('home'))  # Redirect to homepage after logout


@app.route('/index2')
def index2():
    phone = session.get('phone')
    if not phone:
        return redirect(url_for('login'))

    user = users.find_one({'phone': session['phone']})
    if not user:
        return 'User not found'



    return render_template('index2.html',user=user)
# --------------------------------
# ✅ Prediction page

# Load model
MODEL_PATH = "DenseNet169 -17_val_loss.keras"
model = load_model(MODEL_PATH)


# Disease labels
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
          'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
          'Pneumonia', 'Pneumothorax']

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])



import base64
from io import BytesIO
from flask import Flask
from PIL import Image

def to_b64(img_np):
    img_pil = Image.fromarray(img_np.astype('uint8'))
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

app.jinja_env.filters['to_b64'] = to_b64


def preprocess_image(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img / 255.0
    img = (img - mean) / std
    return img


def predict_disease_from_image(image):
    img_resized = cv2.resize(image, (224, 224))
    img_preprocessed = preprocess_image(img_resized)
    predictions = model.predict(np.expand_dims(img_preprocessed, axis=0))
    return predictions


def grad_cam(model, img_array, class_idx, layer_name="conv5_block16_concat"):
    """Generate the GradCAM heatmap for a specific class index and layer."""
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    grad_model = tf.keras.models.Model(
        inputs=[model.input],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        tape.watch(img_array)
        conv_output, predictions = grad_model(inputs=img_array, training=False)
        class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(tf.multiply(conv_output, pooled_grads), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.reduce_max(heatmap)  # Normalize
    return heatmap.numpy()



def generate_gradcam(image, predictions, top_n=3):
    top_n_indices = np.argsort(predictions[0])[-top_n:][::-1]
    top_n_probs = predictions[0][top_n_indices]
    top_n_labels = [labels[i] for i in top_n_indices]

    gradcam_images = []
    for i in range(top_n):
        gradcam = grad_cam(model, np.expand_dims(preprocess_image(image), axis=0), top_n_indices[i])
        gradcam_resized = cv2.resize(gradcam, (224, 224))
        overlay = np.uint8(255 * gradcam_resized)
        heatmap = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255.0

        img_resized_float = np.float32(image) / 255.0
        if len(img_resized_float.shape) == 2:
            img_resized_float = np.repeat(img_resized_float[..., np.newaxis], 3, axis=-1)

        blended_img = 0.5 * img_resized_float + 0.5 * heatmap
        blended_img = np.uint8(255 * blended_img)
        gradcam_images.append((blended_img, top_n_labels[i], top_n_probs[i]))

    return gradcam_images


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    phone = session.get('phone')
    if not phone:
        return redirect(url_for('login'))

    user = users.find_one({'phone': session['phone']})
    if not user:
        return 'User not found'

    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream).convert('RGB')
            image_np = np.array(image)

            predictions = predict_disease_from_image(image_np)
            gradcams = generate_gradcam(image_np, predictions, top_n=6)

            top_diseases = sorted([(labels[i], predictions[0][i]) for i in range(len(labels))],
                                  key=lambda x: x[1], reverse=True)[:6]

            # ✅ Save for report download
            session['top_diseases'] = [(disease, float(prob)) for disease, prob in top_diseases]

            return render_template('prediction.html',
                                   original_image=file.filename,
                                   top_diseases=top_diseases,
                                   gradcams=gradcams)

    # On GET, clear previous predictions
    session.pop('top_diseases', None)

    return render_template('prediction.html', user=user)


# --------------------------------
# ✅ Report Route



from fpdf import FPDF
from flask import send_file

@app.route('/download-report')
def download_report():
    top_diseases = session.get('top_diseases')
    if not top_diseases:
        return redirect(url_for('prediction'))

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Disease Prediction Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Top Predicted Diseases:", ln=True)

    for disease, prob in top_diseases:
        pdf.cell(200, 10, txt=f"{disease}: {prob * 100:.2f}%", ln=True)

    # ✅ Write PDF to memory
    pdf_output = BytesIO()
    pdf_data = pdf.output(dest='S').encode('latin1')
    pdf_output.write(pdf_data)
    pdf_output.seek(0)

    return send_file(pdf_output, download_name="prediction_report.pdf", as_attachment=True)



@app.route('/appointment')
def appointment():
    phone = session.get('phone')
    if not phone:
        return redirect(url_for('login'))

    user = users.find_one({'phone': session['phone']})
    if not user:
        return 'User not found'

    return render_template('appointment.html',user=user)



if __name__ == '__main__':
    app.run(debug=True)
