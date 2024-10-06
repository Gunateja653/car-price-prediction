from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your pre-trained model
model = joblib.load('car_price_model.pkl')

# Sample mapping for car names to numeric codes
car_name_mapping = {
    'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5,
    'Ford': 6, 'Renault': 7, 'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10,
    'Fiat': 11, 'Datsun': 12, 'Jeep': 13, 'Mercedes-Benz': 14,
    'Mitsubishi': 15, 'Audi': 16, 'Volkswagen': 17, 'BMW': 18,
    'Nissan': 19, 'Lexus': 20, 'Jaguar': 21, 'Land': 22, 'MG': 23,
    'Volvo': 24, 'Daewoo': 25, 'Kia': 26, 'Force': 27, 'Ambassador': 28,
    'Ashok': 29, 'Isuzu': 30, 'Opel': 31, 'Peugeot': 32
}

# Mappings for other categorical variables
fuel_name_mapping = {
    'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4
}

seller_type_mapping = {
    'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3
}

transmission_mapping = {
    'Manual': 1, 'Automatic': 2
}

owner_mapping = {
    'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3,
    'Fourth & Above Owner': 4, 'Test Drive Car': 5
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        car_name = request.form['name']
        year = int(request.form['year'])
        km_driven = int(request.form['km_driven'])
        fuel = request.form['fuel']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']
        owner = request.form['owner']
        mileage = float(request.form['mileage'])  # Ensure 'mileage' matches the form input
        engine = float(request.form['engine'])
        max_power = float(request.form['max_power'])
        seats = int(request.form['seats'])

        # Convert car name and other categorical variables to numeric codes
        car_name_code = car_name_mapping.get(car_name, 0)  # Default to 0 if not found
        fuel_name_code = fuel_name_mapping.get(fuel, 0)
        seller_type_code = seller_type_mapping.get(seller_type, 0)
        transmission_code = transmission_mapping.get(transmission, 0)
        owner_code = owner_mapping.get(owner, 0)

        # Input array for the model
        input_data = np.array([[car_name_code, year, km_driven, fuel_name_code,
                                seller_type_code, transmission_code, owner_code,
                                mileage, engine, max_power, seats]])

        # Make prediction using the loaded model
        prediction = model.predict(input_data)

        return render_template('result.html', prediction=prediction)
    
    except KeyError as e:
        return f"KeyError: {str(e)} - Please make sure all form fields are filled."
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
