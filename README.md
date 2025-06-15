HOUSE PRICE PREDICTION MELBOURNE DATASET
  This project predicts house prices in Melbourne using various features like rooms, landsize, house type, etc., using multiple ML models and select best performing one based on RMSE.
PROJECT OVERVIEW:
--->Goal: Predicting house prices based on property features.
--->Tools and Technologies used: Python, Pandas, Scikit-learn, XGBoost, Flask
--->Dataset: Melbourne House Price Prediction Dataset(Kaggle).
FEATURES USED:
---> Rooms, Bathroom, Car
--->Landsizr, BuildingArea, YearBuilt
--->Regionname, CouncilArea
--->Suburb, Distance, Postcode
DATA PREPROCESSING:
-conducted missing value analysis to determine the nature of missingness (e.g.MCAR, MAR)
-Imputed missing values using appropriate central tendency measures(mean/median) based on the distribution of the data.
-Identified and treated outliers using the IQR method.
-Encoded categorical values using label encoding and one-hot encoding where required.
-Applied feature scaling to normalize numerical features for model compatibility.
MODELS USED:
-Linear Regression
-Decision Tree Regressor
-XG Boost Regressor(FINALIZED)
EVALUATION METRIC: RMSE
WEB INTERFACE:
-Developed a user friendly web application using Flask.
-Users can select property attributes (like Region, Rooms, Type, and Method) from dropdown menus populated dynamically from the dataset.
-Upon submission the app preprocess the inputs and passes them to the trained ML model.
-Then model returns an accurate predicted house price based on the selected features.
-The interface is designed to be simple, responsive and easy to use, making it suitable for both technical and non technical users.
USER INTERFACE SNAPSHOTS:
  ![Screenshot 2025-06-15 150713](https://github.com/user-attachments/assets/3a87ffdf-9229-4a8c-ad8b-8f3c111f8afc)
  ![Screenshot 2025-06-15 151024](https://github.com/user-attachments/assets/ef293c16-718d-401c-b6d8-5feac9dca624)




  
