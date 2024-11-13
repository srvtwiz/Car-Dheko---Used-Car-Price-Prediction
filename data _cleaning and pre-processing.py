import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle

file_paths = {'bangalore':'data/bangalore_cars.xlsx','chennai': 'data/chennai_cars.xlsx', 'delhi':'data/delhi_cars.xlsx','hyderabad':'data/hyderabad_cars.xlsx','jaipur':'data/jaipur_cars.xlsx','kolkata':'data/kolkata_cars.xlsx'}
city_dataframes={}

# Loop through each Excel file, load it, and add city column
for city, path in file_paths.items():
    df = pd.read_excel(path,engine='openpyxl')
    df['City'] = city
    city_dataframes[city] = df

# Concatenate all city data into a single DataFrame
combined_df = pd.concat(city_dataframes.values(), ignore_index=True)

print(combined_df)

#a function to extract relevant fields 
def extract_features(row):
    try:
        car_details = ast.literal_eval(row['new_car_detail'])
        car_overview = ast.literal_eval(row['new_car_overview'])
        car_specs = ast.literal_eval(row['new_car_specs'])
    except (ValueError, SyntaxError):
        return pd.Series([None] * 7)
    
    #extract features
    fuel_type = car_details.get('ft', None)
    body_type = car_details.get('bt', None)
    transmission = car_details.get('transmission', None)
    km_driven = car_details.get('km', None)
    model_year = car_details.get('modelYear', None)
    No_owner=car_details.get('ownerNo', None)
    brand=car_details.get('oem',None)
    model=car_details.get('model',None)
    price=car_details.get('price',None)

    
    #extract engine details
    engine = None
    for spec in car_specs.get('top', []):
        if spec.get('key') == 'Engine':
            engine = spec.get('value', None)
            break

    #extract registration year
    Registration_Year=None
    for item in car_overview.get('top', []):
        if item.get('key') == 'Registration Year':
            Registration_Year = item.get('value', None)
            break

    return pd.Series([fuel_type, body_type, transmission, km_driven, model_year, price, engine, brand, model, Registration_Year],
                     index=['Fuel_Type', 'Body_Type', 'Transmission', 'KM_Driven', 'Model_Year', 'Price', 'Engine', 'Brand', 'Model', 'Registration_Year'])

# Apply extraction to each row
extracted_data = combined_df.apply(extract_features, axis=1)

# Combine extracted features with city information
structured_df = pd.concat([extracted_data, combined_df['City']], axis=1)


print(structured_df)

# Convert to numeric by removing commas and converting to int/float
structured_df['KM_Driven'] = structured_df['KM_Driven'].replace({',': ''}, regex=True).astype(int)
structured_df['Engine'] = structured_df['Engine'].str.replace(' CC', '', regex=True).astype(float)

# Extract the year from the registration date
structured_df['Registration_Year'] = structured_df['Registration_Year'].str[-4:]
structured_df['Registration_Year'] = pd.to_numeric(structured_df['Registration_Year'], errors='coerce').astype('Int64')


def convert_price_to_lakh(price):
    # Remove currency symbol and any commas
    price = price.replace('₹', '').replace(',', '').strip()
    
    # Split into amount and unit (Lakh, Crore, etc.)
    parts = price.split()
    if len(parts) == 2:
        amount = float(parts[0])
        unit = parts[1].lower()
        
        # Convert based on unit
        if unit == 'lakh':
            return amount
        elif unit == 'crore':
            return amount * 100
        elif unit == 'thousand':
            return amount / 100  
    return np.nan  # Return NaN if format is unexpected

# Apply conversion function to Price column
structured_df['Price'] = structured_df['Price'].apply(convert_price_to_lakh)

# Convert to numeric
structured_df['Price'] = pd.to_numeric(structured_df['Price'], errors='coerce')

print(structured_df)
print(structured_df.info())

# Fill missing values in 'Registration_Year' with values from 'Model_Year'
structured_df['Registration_Year'] = structured_df['Registration_Year'].fillna(structured_df['Model_Year'])
# Drop rows with missing values in the 'Engine' column
structured_df = structured_df.dropna(subset=['Engine'])

# Fill missing values in 'Price' with median price
median_price = structured_df['Price'].median()
structured_df['Price'] = structured_df['Price'].fillna(median_price)

# Count the number of duplicate rows across all columns
num_duplicates = structured_df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")

# Remove duplicate rows across all columns
structured_df = structured_df.drop_duplicates()

# Count the number of duplicate rows across all columns
num_duplicates = structured_df.duplicated().sum()

print(structured_df)
print(f"Number of duplicate rows: {num_duplicates}")
print(structured_df.info())


# Save DataFrame to CSV file
structured_df.to_csv('cleaned_car_data.csv', index=False)

#pre-processing and encoding
X=structured_df.drop(columns=['Price'])
Y=structured_df['Price']

# Define categorical and numerical columns
categorical_cols = ['Fuel_Type', 'Body_Type', 'Transmission', 'Brand', 'City']
numeric_cols = ['KM_Driven', 'Engine', 'Registration_Year', 'Model_Year']

# Create transformations
preprocessed=ColumnTransformer(transformers=[('num',StandardScaler(),numeric_cols),('cat',OneHotEncoder(),categorical_cols)])

# Transform data
X_preprocessed = preprocessed.fit_transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(X_preprocessed,Y, test_size=0.2, random_state=69)

#train model

model={'Linear Regression':LinearRegression(),'Random Forest':RandomForestRegressor(random_state=69),'Decision Tree': DecisionTreeRegressor(random_state=69),
       'Gradient Boosting': GradientBoostingRegressor(random_state=69)}

for name,model in model.items():
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    print(f"{name} Evaluation:")
    print(f"MAE: {mean_absolute_error(Y_test, Y_pred):.2f}")
    print(f"MSE: {mean_squared_error(Y_test, Y_pred):.2f}")
    print(f"R2 score: {r2_score(Y_test, Y_pred):.2f}\n")

# Hyperparameter tuning for Random Forest using GridSearchCV
rf_params = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}


# Run GridSearchCV on the Random Forest model
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, scoring='r2', cv=5, n_jobs=-1)
grid_search_rf.fit(X_train, Y_train)

# Display the best parameters and the best score from the grid search
print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best R2 score from Grid Search:", grid_search_rf.best_score_)

# Use the best estimator to predict on the test set
best_rf_model = grid_search_rf.best_estimator_
Y_pred_best_rf = best_rf_model.predict(X_test)

# Evaluate the best model on the test data
print("Optimized Random Forest Evaluation:")
print(f"MAE: {mean_absolute_error(Y_test, Y_pred_best_rf):.2f}")
print(f"MSE: {mean_squared_error(Y_test, Y_pred_best_rf):.2f}")
print(f"R2 Score: {r2_score(Y_test, Y_pred_best_rf):.2f}\n")



# Save the best model using pickle
with open('best_rf_model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)

print("The optimized Random Forest model has been saved as 'best_rf_model.pkl'.")

