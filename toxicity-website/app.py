from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Load the trained model
model = tf.keras.models.load_model('models\comment_toxicity_detection.h5')

# Load the tokenizer
with open('models/tokenizer.json', 'r') as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

# Initialize Flask app
app = Flask(__name__)

# Define the toxicity labels
y_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Function to predict toxicity
def predict_toxicity(comment):
    # Tokenize and pad the comment
    sequences = tokenizer.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')

    # Get prediction from the model
    prediction = model.predict(padded_sequence)
    
    # Format the results
    results = {}
    for label, score in zip(y_columns, prediction[0]):
        results[label] = {
            'score': round(float(score), 2),
            'label': 'Yes' if score > 0.5 else 'No'
        }
    return results

# Route to the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the comment submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the comment from the request
        comment = request.form['comment']

        # Check if comment is empty
        if not comment.strip():
            return jsonify({"error": "Please enter a valid comment."}), 400

        # Get prediction results
        prediction_results = predict_toxicity(comment)

        # Return the prediction as JSON response
        return jsonify(prediction_results)

    except Exception as e:
        # Handle unexpected errors
        print(f"Error: {e}")
        return jsonify({"error": "There was an error analyzing the comment."}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
