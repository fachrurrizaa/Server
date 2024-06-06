import pickle
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type"]}})

# Load the pre-trained model and label encoder
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    app.logger.info('Model loaded successfully.')

    decoding_list = [
        'Aamon', 'Akai', 'Aldous', 'Alice', 'Alpha', 'Alucard', 'Angela', 'Argus', 'Arlott', 'Atlas', 'Aulus', 'Aurora',
        'Badang', 'Balmond', 'Bane', 'Barats', 'Baxia', 'Beatrix', 'Belerick', 'Benedetta', 'Brody', 'Bruno', 'Carmilla',
        'Cecilion', 'Change', 'Chou', 'Claude', 'Clint', 'Cyclops', 'Diggie', 'Dyrroth', 'Edith', 'Esmeralda', 'Estes',
        'Eudora', 'Fanny', 'Faramis', 'Floryn', 'Franco', 'Fredrinn', 'Freya', 'Gatotkaca', 'Gloo', 'Gord', 'Granger',
        'Grock', 'Guinevere', 'Gusion', 'Hanabi', 'Hanzo', 'Harith', 'Harley', 'Hayabusa', 'Helcurt', 'Hilda', 'Hylos',
        'Irithel', 'Jawhead', 'Johnson', 'Joy', 'Julian', 'Kadita', 'Kagura', 'Kaja', 'Karina', 'Karrie', 'Khaleed',
        'Khufra', 'Kimmy', 'Lancelot', 'Lapu-Lapu', 'Layla', 'Leomord', 'Lesley', 'Ling', 'Lolita', 'Lunox', 'Luo Yi',
        'Lylia', 'Martis', 'Masha', 'Mathilda', 'Melissa', 'Minotaur', 'Minsitthar', 'Miya', 'Moskov', 'Nana', 'Natalia',
        'Natan', 'Odette', 'Paquito', 'Pharsa', 'Phoveus', 'Popol and Kupa', 'Rafaela', 'Roger', 'Ruby', 'Saber', 'Selena',
        'Silvanna', 'Sun', 'Terizla', 'Thamuz', 'Tigreal', 'Uranus', 'Vale', 'Valentina', 'Valir', 'Vexana', 'Wanwan',
        'X.Borg', 'Xavier', 'Yi Sun-shin', 'Yin', 'Yu Zhong', 'Yve', 'Zhask', 'Zilong'
    ]
    encoder = LabelEncoder()
    encoder.fit(decoding_list)
    app.logger.info('Label encoder loaded and fitted successfully.')
except Exception as e:
    app.logger.error('Error loading model or label encoder:', exc_info=e)

@app.route('/')
def index():
    return 'Hello, World!'

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        selected_heroes = data.get('selectedHeroes')

        app.logger.debug('Received data: %s', selected_heroes)

        # Encode the hero names
        encoded_heroes = encoder.transform(selected_heroes)

        # Convert to DataFrame
        input_df = pd.DataFrame([encoded_heroes])

        app.logger.debug('Encoded heroes: %s', encoded_heroes)
        
        # Use the model to make predictions
        prediction = model.predict(input_df)
        
        # Determine the win status
        winStatus = 'Menang' if prediction[0] == 1 else 'Kalah'
        app.logger.info('Prediction successful: %s', winStatus)
        
        return jsonify({'result': winStatus})
    except Exception as e:
        app.logger.error('Error during prediction:', exc_info=e)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
