Customer Churn Prediction API

Introduction

The Customer Churn Prediction API is a simple and practical machine learning project built using Flask. Its main goal is to predict whether a customer is likely to leave (churn) based on their details.

The system uses a Decision Tree model along with a preprocessing pipeline to handle raw input data. It supports both real-time predictions through an API and batch processing for large datasets. The project is designed in a clean and structured way so that it can be easily understood, tested, and extended.


Project Structure

The project is organized into different folders and files, each serving a specific purpose:

customer-churn-api/

* app/

  * **init**.py → Initializes the app as a Python package
  * main.py → Contains the Flask API and endpoints
  * model.pkl → Trained Decision Tree model
  * transformer.pkl → Preprocessing pipeline
  * utils.py → Helper functions for loading and preprocessing

* test_data/

  * sample_input.json → Example input for testing API
  * all_customers.csv → Dataset for batch scoring

* logs/

  * batch_log.txt → Stores logs from batch runs

* batch.py → Script for batch prediction

* requirements.txt → List of dependencies

* README.md → Documentation


## Steps to Execute

 1. Install Dependencies

First, install all the required Python libraries:

pip install -r requirements.txt

 2. Run the API

Move to the project directory and start the server:

cd customer-churn-api
python -m app.main

The API will start running at:
http://localhost:8000

3. Test the API

You can test the API using tools like Postman or curl.

POST /predict

Send a JSON request like this:

{
"customer": {
"gender": "Female",
"SeniorCitizen": 0,
"Partner": "Yes",
"tenure": 12,
"MonthlyCharges": 29.85,
"TotalCharges": 358.20,
"Contract": "Month-to-month"
}
}

Response:

{
"churn_probability": 0.83,
"churn_prediction": "Yes"
}

---

GET /health

This endpoint checks if the server is running:

{
"status": "ok"
}

 4. Run Batch Scoring

To process multiple customers at once:

python batch.py --input test_data/all_customers.csv

This will generate:

* scored_customers.csv → Contains predictions for all customers
* logs/batch_log.txt → Contains logs like total requests and failures

 How It Works

 1. Input Handling

The system accepts customer data either:

* Through the API (single input)
* Through a CSV file (batch input)

 2. Data Preprocessing

Before making predictions, the data is passed through a preprocessing pipeline (transformer.pkl). This step includes:

* Encoding categorical variables
* Scaling numerical values
* Ensuring the data is in the correct format

 3. Model Prediction

The processed data is then given to the trained Decision Tree model (model.pkl).
The model calculates the probability of churn and classifies the customer as "Yes" or "No".

 4. Output Generation

The system returns:

* churn_probability → Likelihood of churn
* churn_prediction → Final decision

 5. Batch Processing Logic

In batch mode:

* The CSV file is read
* Each record is processed
* Predictions are added as new columns
* Logs are created to track performance

 6. Maintenance and Monitoring

To keep the system reliable:

* Retraining is done regularly (monthly or when performance drops)
* Data drift and model drift are monitored
* Models are versioned using timestamps
* Changes are documented for tracking


## Final Summary

This project is a complete example of deploying a machine learning model in a real-world scenario. It includes:

* API-based real-time predictions
* Batch processing for scalability
* Clean project structure
* Monitoring and maintenance strategies

Overall, it shows how a machine learning model can be turned into a usable product.

