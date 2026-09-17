HeartCheck - Flask Heart Disease Prediction Web App

Dataset:
- heart.csv
- Rows: 918
- Input features: 11
- Target: HeartDisease

Machine learning:
- Random Forest Classifier
- One-hot encoding for categorical features
- Standard scaling for numeric features
- 5-fold stratified ROC-AUC during model evaluation
- Mean cross-validation ROC-AUC on the supplied dataset: 0.9284

Project structure:
HeartHealthChecker/
  app.py
  train_model.py
  heart.csv
  heart-disease-model.pkl
  requirements.txt
  templates/
    index.html
  static/
    css/
      style.css

Run:
1. Open a terminal in this folder.
2. Install dependencies:
   pip install -r requirements.txt
3. Start the app:
   python app.py
4. Open:
   http://127.0.0.1:5000/

Retrain:
python train_model.py

Important:
This application is an educational machine-learning demonstration. Its prediction is not a medical diagnosis and should not replace evaluation by a qualified healthcare professional.
