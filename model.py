import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import pickle


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import mplcursors




pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:,.3f}".format)




df = pd.read_csv("final_data_house_sales2025.csv", encoding="latin1")

df = df.drop(columns=["LSOA"])

df["Type_raw"] = df["Type"]

df = pd.get_dummies(df, columns=["Type"], drop_first=True)



X = df.drop(columns=["Paid Price"])
y = df["Paid Price"]

postcodes = X["Postcode"]
addresses = X["Address"]
types_raw = df["Type_raw"]

X = X.drop(columns=["Postcode", "Address", "Type_raw"])



# Train-test split

X_train, X_test, \
y_train, y_test, \
pc_train, pc_test, \
addr_train, addr_test, \
type_train, type_test = train_test_split(
    X, y, postcodes, addresses, types_raw,
    test_size=0.2,
    random_state=88
)


#models
#model = LinearRegression()
#model = Ridge(alpha=50)
model = RandomForestRegressor(n_estimators=100,max_depth=None,min_samples_split=5,min_samples_leaf=2,random_state=50)
#model = GradientBoostingRegressor(n_estimators=300,learning_rate=0.1, max_depth=4,random_state=50,min_samples_split=5,min_samples_leaf=2)
#model = MLPRegressor( hidden_layer_sizes=(100, 50), activation='relu',solver='adam',learning_rate_init=0.01, max_iter=500,  random_state=50, alpha = 0.001)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#model.fit(X_train_scaled, y_train)
#y_pred = model.predict(X_test_scaled)



# Residual analysis


residuals = y_pred - y_test
abs_error = np.abs(residuals)
pct_error = residuals / y_test * 100

results = pd.DataFrame({
    "Address": addr_test,
    "Postcode": pc_test,
    "Type": type_test,
    "ActualPrice": y_test,
    "PredictedPrice": y_pred,
    "Residual": residuals,
    "AbsError": abs_error,
    "PctError": pct_error
})



# Overall metrics


overall_metrics = pd.DataFrame({
    "RMSE": [np.sqrt(mean_squared_error(y_test, y_pred))],
    "MAE": [mean_absolute_error(y_test, y_pred)],
    "R2": [r2_score(y_test, y_pred)]
})

print("\n=== OVERALL MODEL PERFORMANCE===\n")
print(overall_metrics)



# Coefficients

#coefficients = pd.Series(
#    model.coef_,
#    index=X.columns
#).sort_values()

#print("\n===COEFFICIENTS===\n")
#print(coefficients)



# Performance  by property type


type_metrics = (
    results
    .groupby("Type")
    .apply(lambda g: pd.Series({
        "Count": len(g),
        "RMSE": np.sqrt(mean_squared_error(g["ActualPrice"], g["PredictedPrice"])),
        "MAE": mean_absolute_error(g["ActualPrice"], g["PredictedPrice"]),
        "MeanResidual": g["Residual"].mean(),
        "MeanPctError": g["PctError"].mean()
    }))
    .sort_values("RMSE")
)

print("\n=== PERFORMANCE BY PROPERTY TYPE ===\n")
print(type_metrics)



# Price bands


results["PriceBand"] = pd.qcut(
    results["ActualPrice"],
    q=4,
    labels=["Low", "Mid-Low", "Mid-High", "High"]
)

price_band_metrics = (
    results
    .groupby("PriceBand")
    .apply(lambda g: pd.Series({
        "Count": len(g),
        "RMSE": np.sqrt(mean_squared_error(g["ActualPrice"], g["PredictedPrice"])),
        "MAE": mean_absolute_error(g["ActualPrice"], g["PredictedPrice"]),
        "MeanResidual": g["Residual"].mean(),
        "MeanPctError": g["PctError"].mean()
    }))
)

print("\n=== PERFORMANCE BY PRICE BAND ===\n")
print(price_band_metrics)





results["ValueScore"] = results["Residual"] / results["ActualPrice"]

top_steals = results.sort_values("ValueScore", ascending=False).head(20)
worst_areas = results.sort_values("ValueScore").head(20)

display_cols = [
    "Address", "Postcode", "Type",
    "ActualPrice", "PredictedPrice",
    "Residual", "ValueScore"
]

print("\n=== TOP 20 UNDERPRICED PROPERTIES ===\n")
print(top_steals[display_cols])

print("\n=== TOP 20 OVERPRICED PROPERTIES ===\n")
print(worst_areas[display_cols])




# Scatter plot (< £1.5m)



PRICE_CAP = 1_500_000
mask = y_test <= PRICE_CAP

fig, ax = plt.subplots(figsize=(8, 8))

scatter = ax.scatter(
    y_test[mask],
    y_pred[mask],
    alpha=0.6
)

ax.plot([-100000, PRICE_CAP], [-100000, PRICE_CAP])

ax.set_xlim(-100000, PRICE_CAP)
ax.set_ylim(-100000, PRICE_CAP)

ax.set_xlabel("Actual Price (£ millions)")
ax.set_ylabel("Predicted Price (£ millions)")
ax.set_title("Predicted vs Actual Prices (≤ £1.5m)")

ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))

cursor = mplcursors.cursor(scatter, hover=True)

@cursor.connect("add")
def on_add(sel):
    idx = sel.index
    actual = y_test[mask].iloc[idx]
    predicted = y_pred[mask][idx]
    address = addr_test[mask].iloc[idx]

    sel.annotation.set_text(
        f"{address}\n"
        f"Actual: £{actual:,.0f}\n"
        f"Predicted: £{predicted:,.0f}"
    )

plt.tight_layout()
plt.show()



def predict_house_price(
    floor_area_m2,
    property_type,
    employment,
    health,
    crime,
    income,
    education,
    model,
    feature_columns
):

    X_new = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    X_new["total_floor_area_m2"] = floor_area_m2
    X_new["Employment"] = employment
    X_new["Health"] = health
    X_new["Crime"] = crime
    X_new["Income"] = income
    X_new["Education"] = education

    if property_type in ["F", "S", "T"]:
        X_new[f"Type_{property_type}"] = 1
    elif property_type == "D":
        pass
    else:
        raise ValueError("property_type must be one of: 'D', 'F', 'S', 'T'")

    return model.predict(X_new)[0]



# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)