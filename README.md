# Comment Toxicity Detection System

## Project Overview

This project implements a machine learning-powered Comment Toxicity Detection system designed to automatically identify and classify toxic comments across online platforms. By leveraging advanced natural language processing (NLP) techniques, the system helps moderators and platform administrators maintain a healthy and respectful online environment.

## 🌟 Key Features

- **Intelligent Toxicity Classification**: Detect various forms of harmful content
- **Machine Learning Model**: Trained in Jupyter Notebook
- **TensorFlow Lite Optimization**: Model compressed for efficient deployment
- **Flask Web Application**: Easy-to-use interface for toxicity analysis
- **Multi-Category Detection**: Identifies different types of toxic content

## 🛠 Technology Stack

- **Programming Language**: Python
- **Machine Learning**: 
  - TensorFlow
  - Keras
  - Jupyter Notebook
- **Model Optimization**: TensorFlow Lite
- **Web Framework**: Flask
- **Data Processing**: Pandas, NumPy
- **Text Processing**: NLTK, SpaCy

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Setup Steps

1. Clone the repository
```bash
git clone https://github.com/TaneshG13/Comment-Toxicity-detection.git
cd Comment-Toxicity-detection
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Running the Application

### Model Training
The model is trained in the Jupyter Notebook located in the `notebooks/` directory:
- Model is saved as an `.h5` file
- Converted to TensorFlow Lite (`.tflite`) for optimized inference

### Running Flask Application
```bash
python app.py
```

## 🖼 Screenshots

### 1. Home Page
![Non-Toxic Comment](/Screenshots/sc1.png)
![Toxic Comment](/Screenshots/sc3.png)

### 2. Toxicity Analysis Result
![Non-Toxic Comment Result](/Screenshots/sc2.png)
![Toxic Comment Result](/Screenshots/sc4.png)

## 📊 Model Performance

- **Test Accuracy**: 99%
- **Precision**: 0.75 (weighted avg)
- **Recall**: 0.61 (weighted avg)
- **F1 Score**: 0.67 (weighted avg)

## 🔍 Toxicity Categories Detected

1. Hate Speech
2. Bullying
3. Profanity
4. Threat
5. Explicit Content

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

### 💡 Disclaimer

This tool is designed to assist in content moderation and should be used as a supportive tool, not as the sole decision-making mechanism for content filtering.
