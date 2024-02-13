from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import cv2
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the pre-trained model
train_folder = "D:\\KULIAH\\SEMESTER 5\\TEKCERTAN\\Proyek\\CLASIFICATION SAMPAH\\train"
image_paths = glob(os.path.join(train_folder, "*", "*.jpg"))

features = []
labels = []

for path in image_paths:
    label = os.path.basename(os.path.dirname(path))
    labels.append(label)

    img = cv2.imread(path)
    img = cv2.resize(img, (100, 100))
    features.append(img.flatten())

features = np.array(features)
labels = np.array(labels)

X_train, _, y_train, _ = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


def predict_image_category(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    img = img.flatten().reshape(1, -1)

    prediction = model.predict(img)

    return prediction[0]


# Configure the upload folder
app.config['UPLOAD_FOLDER'] = 'D:\\KULIAH\\SEMESTER 5\\TEKCERTAN\\Proyek\\CLASIFICATION SAMPAH\\test'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image_file']

        if image_file:
            # Securely save the uploaded file to a temporary location
            filename = secure_filename(image_file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(temp_path)

            # Perform prediction on the uploaded image
            predicted_category = predict_image_category(rf_model, temp_path)

            # Read the uploaded image
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = cv2.imread(img_path)

            # Display the uploaded image and prediction
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Image to Classify - Classified Category: {}".format(predicted_category))
            plt.show()

            return render_template('result.html', image_path=filename, predicted_category=predicted_category)


if __name__ == '__main__':
    app.run(debug=True)
