# Depression Prediction Web Application

This web application predicts the likelihood of depression based on user inputs. It uses a machine learning model trained on a dataset of student responses to various questions related to their mental health and lifestyle.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model and Data](#model-and-data)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ZeinebTek/Depression-Predictor
    cd depression-prediction
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Ensure the model and encoders are in the `assets` directory:
    - `best_model.joblib`
    - `feature_encoders.joblib`
    - `scaler.joblib`
    - `target_encoder.joblib`

## Usage

1. Run the Flask application:
    ```sh
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000/`.

3. Fill out the form with your details and submit to get the prediction.

## Model and Data

- The model is trained on the `Depression Student Dataset.csv` file located in the `model` directory.
- The Jupyter notebook `depression_student_classification.ipynb` contains the code for training the model.
- The trained model and encoders are saved as `.joblib` files in the `assets` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
