from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load and train the model on server startup
heart_data = pd.read_csv('heart.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Apply feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVC model
clf1 = SVC(kernel="linear")
clf1.fit(X_train, Y_train)

# Calculate accuracy on the test set
accuracy = accuracy_score(Y_test, clf1.predict(X_test)) * 100  # Accuracy in percentage

# Utility function to make predictions
def predict_heart_disease(model, input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Apply the same scaling to the input data
    input_data_scaled = scaler.transform(input_data_reshaped)
    
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Main view to render the index page
def index(request):
    return render(request, 'index.html')

# Prediction view
def predict(request):
    try:
        if 'csv_file' in request.FILES:
            csv_file = request.FILES['csv_file']
            df = pd.read_csv(csv_file)

            selected_patient_data = json.loads(request.POST.get('selected_patient_data', '[]'))
            logging.info("Selected Patient Data: %s", selected_patient_data)

            if len(selected_patient_data) != X.shape[1]:
                expected_count = X.shape[1]
                return render(request, 'result.html', {
                    'error': f'Selected patient data must have {expected_count} features.'
                })

            pred = predict_heart_disease(clf1, selected_patient_data)
            result_text = 'The Person has Heart Disease' if pred == 1 else 'The Person does not have Heart Disease'
            prediction_prob = clf1.decision_function(scaler.transform(np.asarray(selected_patient_data).reshape(1, -1)))[0]

            logging.info("Prediction Result Text: %s", result_text)

            # Pass prediction result, model accuracy, and additional insights to the template
            return render(request, 'result.html', {
                'result': pred,
                'accuracy': accuracy,
                'prediction_prob': prediction_prob,
                'result_text': result_text
            })

        return render(request, 'result.html', {'error': 'No file uploaded.'})

    except Exception as e:
        logging.error("Error: %s", str(e))
        return render(request, 'result.html', {'error': str(e)})

# Preview CSV data
def preview(request):
    try:
        df = pd.read_csv("heart.csv")
        df = df[:10]  # Preview the first 10 rows
        json_records = df.reset_index().to_json(orient='records')
        arr = json.loads(json_records)
        context = {'df': arr}
        return render(request, 'preview.html', context)
    except FileNotFoundError:
        return render(request, 'preview.html', {'error': 'CSV file not found.'})
    except Exception as e:
        logging.error("Error: %s", str(e))
        return render(request, 'preview.html', {'error': str(e)})
