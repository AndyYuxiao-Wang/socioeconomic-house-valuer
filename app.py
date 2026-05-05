import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load main dataset for reference (house data)
df = pd.read_csv("final_data_house_sales2025.csv", encoding="latin1")
df = df.drop(columns=["LSOA"])
df["Type_raw"] = df["Type"]
df = pd.get_dummies(df, columns=["Type"], drop_first=True)

# Load socioeconomic data
socio_data = pd.read_csv("finalData.csv")

print(socio_data.columns.tolist())

app = Flask(__name__)



def preprocess_input(data):
    """
    data: dict with keys ['LSOA21CD', 'Type', 'Floor Area']
    """
    # Use the same columns as training
    feature_columns = [
        "total_floor_area_m2", "Employment", "Health", "Crime",
        "Income", "Education", "Type_F", "Type_S", "Type_T"
    ]

    # Start with zeros
    df_input = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    # Map numeric features
    df_input["total_floor_area_m2"] = float(data["Floor Area"])


    row = socio_data[socio_data['LSOA21CD'] == data['LSOA21CD']]
    if not row.empty:
        df_input["Income"] = row['Total annual income.Total annual income (£)'].values[0]
        df_input["Education"] = row['202425_local_authority_district.attainment8_average'].values[0]
        df_input["Crime"] = row['Crime Decile (where 1 is most deprived 10% of LSOAs)'].values[0]
        df_input["Employment"] = row.get("Employment", 0) if "Employment" in row.columns else 0
        df_input["Health"] = row.get("Health", 0) if "Health" in row.columns else 0
    else:
        df_input[["Income", "Education", "Crime", "Employment", "Health"]] = 0

    prop_type_map = {"Detached": "Type_D", "Semi-Detached": "Type_S", "Terrace": "Type_T", "Flat": "Type_F"}
    type_col = prop_type_map.get(data["Type"])
    if type_col and type_col in feature_columns:
        df_input[type_col] = 1

    # Scale features
    X_scaled = scaler.transform(df_input)
    return X_scaled



@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X_scaled = preprocess_input(data)
    pred_price = model.predict(X_scaled)[0]

    # Recommended price range ±10%
    low = round(pred_price * 0.9, 0)
    high = round(pred_price * 1.1, 0)

    # Socioeconomic info
    row = socio_data[socio_data['LSOA21CD'] == data['LSOA21CD']]
    if not row.empty:
        # Get latitude and longitude - adjust column names as needed
        lat = float(row["Latitude"].values[0]) if "Latitude" in row.columns else 51.505
        lon = float(row["Longitude"].values[0]) if "Longitude" in row.columns else -0.09

        socio_info = {
            "Income": int(row['Total annual income.Total annual income (£)'].values[0]),
            "Education": float(row['202425_local_authority_district.attainment8_average'].values[0]),
            "Crime": int(row['Crime Decile (where 1 is most deprived 10% of LSOAs)'].values[0]),
            "lat": lat,
            "lon": lon
        }
    else:
        socio_info = {
            "Income": 0,
            "Education": 0.0,
            "Crime": 0,
            "lat": 51.505,
            "lon": -0.09
        }


    result = {
        "predicted_price": float(round(pred_price, 0)),
        "recommended_range": [float(low), float(high)],
        "socio": socio_info
    }
    return jsonify(result)
socio_dict = {}
for _, row in socio_data.iterrows():
    socio_dict[row['LSOA21CD']] = {
        'Income': int(row.get('Total annual income.Total annual income (£)', 0)),
        'Education': float(row.get('202425_local_authority_district.attainment8_average', 0)),
        'Crime': int(row.get('Crime Decile (where 1 is most deprived 10% of LSOAs)', 0)),
        'lat': float(row.get('Latitude', 51.505)) if 'Latitude' in row else 51.505,
        'lon': float(row.get('Longitude', -0.09)) if 'Longitude' in row else -0.09
    }

@app.route('/lsoa-data')
def get_lsoa_data():
    """Return all LSOA socioeconomic data for the map"""
    #Calculate min/max for scaling
    income_values = [v['Income'] for v in socio_dict.values() if v['Income'] > 0]
    education_values = [v['Education'] for v in socio_dict.values() if v['Education'] > 0]
    crime_values = [v['Crime'] for v in socio_dict.values() if v['Crime'] > 0]

    return jsonify({
        'socio_data': socio_dict,
        'stats': {
            'income': {
                'min': min(income_values) if income_values else 0,
                'max': max(income_values) if income_values else 1
            },
            'education': {
                'min': min(education_values) if education_values else 0,
                'max': max(education_values) if education_values else 1
            },
            'crime': {
                'min': min(crime_values) if crime_values else 1,
                'max': max(crime_values) if crime_values else 10
            }
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
