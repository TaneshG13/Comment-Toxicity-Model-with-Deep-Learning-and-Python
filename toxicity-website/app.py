import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Load the TFLite model
try:
    model_path = os.path.join('models', 'comment_toxicity_detection.tflite')
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    raise RuntimeError(f"Error loading the TFLite model: {e}")

# Load the tokenizer
try:
    tokenizer_path = os.path.join('models', 'tokenizer.json')
    with open(tokenizer_path, 'r') as f:
        tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)
except Exception as e:
    raise RuntimeError(f"Error loading the tokenizer: {e}")

# Define the toxicity labels
y_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Function to predict toxicity
def predict_toxicity(comment):
    try:
        # Tokenize and pad the comment
        sequences = tokenizer.texts_to_sequences([comment])
        padded_sequence = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
        input_data = np.array(padded_sequence, dtype=np.float32)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data[0]

        # Format the results
        results = {}
        for label, score in zip(y_columns, prediction):
            results[label] = {
                'score': round(float(score), 2),
                'label': 'Yes' if score > 0.5 else 'No'
            }
        return results

    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")

# Route to the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the comment submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the comment from the request
        comment = request.form.get('comment', '').strip()

        # Check if the comment is empty
        if not comment:
            return jsonify({"error": "Please enter a valid comment."}), 400

        # Get prediction results
        prediction_results = predict_toxicity(comment)

        # Return the prediction as JSON response
        return jsonify(prediction_results)

    except Exception as e:
        # Log the error and return a response
        print(f"Error: {e}")
        return jsonify({"error": "There was an error analyzing the comment."}), 500

# Run the Flask app
if __name__ == '__main__':
    # Get the PORT environment variable for deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
