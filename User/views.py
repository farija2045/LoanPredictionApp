from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template import loader
import pickle
import numpy as np
import os
from django.conf import settings

# Load the models once when the server starts
try:
    with open(os.path.join(settings.BASE_DIR, 'User', 'Loan_pred_model', 'label_encoder.pkl'), 'rb') as le_file:
        label_encoder = pickle.load(le_file)
except FileNotFoundError:
    raise FileNotFoundError("The file 'label_encoder.pkl' was not found in 'User/Loan_pred_model'.")
except pickle.UnpicklingError:
    raise ValueError("The file 'label_encoder.pkl' is corrupt or incompatible.")

try:
    with open(os.path.join(settings.BASE_DIR, 'User', 'Loan_pred_model', 'rf_model.tuned.pkl'), 'rb') as rf_file:
        rf_model = pickle.load(rf_file)
except FileNotFoundError:
    raise FileNotFoundError("The file 'rf_model.tuned.pkl' was not found in 'User/Loan_pred_model'.")
except pickle.UnpicklingError:
    raise ValueError("The file 'rf_model.tuned.pkl' is corrupt or incompatible.")

try:
    with open(os.path.join(settings.BASE_DIR, 'User', 'Loan_pred_model', 'scaler.pkl'), 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    raise FileNotFoundError("The file 'scaler.pkl' was not found in 'User/Loan_pred_model'.")
except pickle.UnpicklingError:
    raise ValueError("The file 'scaler.pkl' is corrupt or incompatible.")

try:
    with open(os.path.join(settings.BASE_DIR, 'User', 'Loan_pred_model', 'tfidf_vectorizer.pkl'), 'rb') as tfidf_file:
        tfidf_vectorizer = pickle.load(tfidf_file)
except FileNotFoundError:
    raise FileNotFoundError("The file 'tfidf_vectorizer.pkl' was not found in 'User/Loan_pred_model'.")
except pickle.UnpicklingError:
    raise ValueError("The file 'tfidf_vectorizer.pkl' is corrupt or incompatible.")
def predict(request):
    # Example: Get input data from the request (e.g., query parameters)
    text_input = request.GET.get('text', '')
    numeric_input = [float(request.GET.get('feature1', 0)), float(request.GET.get('feature2', 0))]

    # Preprocess the text input using the TF-IDF vectorizer
    text_features = tfidf_vectorizer.transform([text_input])

    # Scale the numeric input
    numeric_features = scaler.transform([numeric_input])

    # Combine text and numeric features
    combined_features = np.hstack((text_features.toarray(), numeric_features))

    # Make a prediction using the random forest model
    prediction = rf_model.predict(combined_features)

    # Decode the prediction using the label encoder
    decoded_prediction = label_encoder.inverse_transform(prediction)

    # Return the prediction as a JSON response
    return JsonResponse({'prediction': decoded_prediction[0]})

def user(request):
    template = loader.get_template('user.html')
    return HttpResponse(template.render())
