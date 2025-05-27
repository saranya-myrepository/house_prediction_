from flask import Flask, render_template, request
import pandas as pd
import joblib
import mysql.connector
from xgboost import XGBRegressor

app = Flask(__name__)

# Friendly dropdown options with numeric codes as keys
dropdown_options = {
    "Regionname": {
        0: "West", 1: "North West", 2: "North", 3: "North East",
        4: "East", 5: "South East", 6: "South", 7: "South West", 8: "Central"
    },
    "Rooms": sorted([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]),
    "Type": {
        0: "House, cottage, villa, semi, terrace",
        1: "Unit, duplex",
        2: "Townhouse",
        3: "Development site",
        4: "Other residential"
    },
    "Method": {
        0: "S - property sold",
        1: "SP - property sold prior",
        2: "PI - property passed in",
        3: "PN - sold prior not disclosed",
        4: "SN - sold not disclosed",
        5: "NB - no bid",
        6: "VB - vendor bid",
        7: "W - withdrawn prior to auction",
        8: "SA - sold after auction",
        9: "SS - sold after auction price not disclosed",
        10: "N/A - price or highest bid not available"
    }
}

input_columns = ['Regionname', 'Rooms', 'Type', 'Method']

# Load models
models = {
    "XGBoost": XGBRegressor()
}
models["XGBoost"].load_model("models/xgboost_model.json")

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="house_price_db"
    )

@app.route('/', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # Convert dropdown selected string values to int codes
        input_data = {
            'Regionname': int(request.form['Regionname']),
            'Rooms': int(request.form['Rooms']),
            'Type': int(request.form['Type']),
            'Method': int(request.form['Method'])
        }

        df_input = pd.DataFrame([input_data])

        # Ensure numeric format
        for col in input_columns:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce')

        # One-hot encoding
        df_encoded = pd.get_dummies(df_input)
        model = models["XGBoost"]
        model_columns = model.feature_names_in_

        for col in model_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[model_columns]

        # Predict
        prediction = model.predict(df_encoded)[0]
        prediction_py = float(prediction)  # Convert from numpy to native float

        # Save to DB (ModelUsed removed)
        conn = get_db_connection()
        cursor = conn.cursor()
        insert_query = """
            INSERT INTO predictions (Regionname, Rooms, Type, Method, PredictedPrice)
            VALUES (%s, %s, %s, %s, %s)
        """
        values = (
            input_data['Regionname'],
            input_data['Rooms'],
            input_data['Type'],
            input_data['Method'],
            round(prediction_py, 2)
        )
        cursor.execute(insert_query, values)
        conn.commit()
        cursor.close()
        conn.close()

        return render_template('success.html', price=round(prediction_py, 2), model="XGBoost")

    return render_template('form.html', dropdowns=dropdown_options, models=list(models.keys()))

if __name__ == '__main__':
    app.run(debug=True)
