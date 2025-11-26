## **Customer Lifetime Value (CLV) Prediction**

This project provides a Streamlit dashboard to predict Customer Lifetime Value (CLV) using a Linear Regression model. The app allows users to input customer attributes, view CLV predictions, understand feature impact, and compare customers against a dataset.

Features:
CLV Prediction
Users can enter key customer metrics such as recency, frequency, average order value, order variability, product diversity, and purchase gaps. The model generates an estimated short-term CLV.

Feature Breakdown
A contribution chart explains how each feature influences the predicted CLV, helping users understand which behaviors increase or decrease customer value.

Dataset Comparison
If Online Retail.xlsx or Online Retail.csv is available, the app computes features from the dataset and shows where the predicted CLV stands relative to other customers. Otherwise, a synthetic sample is used.

Model Auto-Loading
The app automatically:
1)Loads model.pkl if present
2)Trains a model using the dataset if available
3)Uses default coefficients as a fallback
(The repository includes a pre-trained Linear Regression model (model.pkl) that loads automatically.)

Download Results
Predictions can be exported as Excel or CSV files depending on installed packages.

Running the App
Install the required libraries from the requirements,txt file.

Launch the dashboard:  "streamlit run app.py"

Model Training (Optional)
To save your own model:
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)
Place model.pkl in the same directory as app.py. The app will use it automatically.

Use Cases:
This tool can support customer segmentation, revenue forecasting, retention analysis, and marketing planning.

Below are the images of the dashboard:
<img width="1600" height="747" alt="image" src="https://github.com/user-attachments/assets/0260f0b3-d27b-4ad7-80a3-80f38badabf3" />
<img width="1600" height="782" alt="image" src="https://github.com/user-attachments/assets/ac1e9e78-6ca3-499a-9725-7e9f63c0d834" />
<img width="1335" height="300" alt="Screenshot 2025-11-26 at 8 13 28 PM" src="https://github.com/user-attachments/assets/6c7af649-1d2f-46dc-a17d-495a8b72c4a0" />
<img width="1457" height="712" alt="Screenshot 2025-11-26 at 8 13 15 PM" src="https://github.com/user-attachments/assets/dd672bea-39f5-456d-9fe0-4858abfead61" />

