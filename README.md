# Car Evaluation Model API

This repository contains a Flask API for predicting car evaluations based on a machine learning model. The model used is Naive Bayes, trained on a dataset containing car attributes.

## Getting Started

These instructions will help you set up and run the API on your local machine.

### Prerequisites

Make sure you have Python installed on your system. You will also need the following libraries:

- Flask
- scikit-learn
- pandas
- numpy
- flask_cors

You can install the required libraries using pip:

```bash
pip install Flask scikit-learn pandas numpy flask_cors
```

### Running the API

Follow these steps to run the API:

1. **Clone the Repository:**

   ```bash
   git clone <repository-link>
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd car-evaluation-api
   ```

3. **Run the Flask Server:**

   ```bash
   python app.py
   ```

   The server will start running locally on port 5000.

### Usage

Once the server is running, you can send POST requests to the `/predict` endpoint with car attributes in JSON format to get predictions.

Example using cURL:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"0": "vhigh", "1": "vhigh", "2": "2", "3": "2", "4": "small", "5": "low"}' http://localhost:5000/predict
```

Replace the attribute values with your own data.

### Dataset

The dataset used for training the model can be found in the `archive/car_evaluation.csv` folder. It contains car attributes and corresponding evaluations.

## API Endpoints

### `/predict`

- **Method:** POST
- **Description:** Predicts the car evaluation based on provided attributes.
- **Request Body:** JSON object containing car attributes.
- **Response:** JSON string representing the predicted evaluation.

## Testing with Frontend

Alternatively, you can use the [front-end repository]([https://github.com/your-front-end-repo](https://github.com/ticflay/carevaluationfront)) to test the API. Follow the instructions in that repository to set up the front-end and replace the API URL with your local server URL.

Make sure to replace `<repository-link>` with the actual repository link and `your-front-end-repo` with the URL of your front-end repository.
