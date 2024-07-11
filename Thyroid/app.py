from flask import Flask, render_template, request
import pandas as pd
import pickle   

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

# Define columns for one-hot encoding
columns_to_encode = [
    'sex', 'on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant', 
    'thyroid_surgery', 'I131_treatment', 'query_on_thyroxine', 'query_hypothyroid', 
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 
    'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured'
]

# Assuming you have a list of all feature names used during training
feature_names = [
    'age', 'TSH', 'TT4', 'T4U', 'FTI', 'T3',
    'sex_F', 'sex_M', 'on_thyroxine_f', 'on_thyroxine_t', 
    'on_antithyroid_medication_f', 'on_antithyroid_medication_t', 'sick_f', 'sick_t', 
    'pregnant_f', 'pregnant_t', 'thyroid_surgery_f', 'thyroid_surgery_t', 
    'I131_treatment_f', 'I131_treatment_t', 'query_on_thyroxine_f', 'query_on_thyroxine_t', 
    'query_hypothyroid_f', 'query_hypothyroid_t', 'query_hyperthyroid_f', 'query_hyperthyroid_t', 
    'lithium_f', 'lithium_t', 'goitre_f', 'goitre_t', 'tumor_f', 'tumor_t', 
    'hypopituitary_f', 'hypopituitary_t', 'psych_f', 'psych_t', 
    'TSH_measured_f', 'TSH_measured_t', 'T3_measured_f', 'T3_measured_t', 
    'TT4_measured_f', 'TT4_measured_t', 'T4U_measured_f', 'T4U_measured_t', 
    'FTI_measured_f', 'FTI_measured_t'
]

@app.route("/", methods=['GET'])
def home():
    return render_template('home.html')

# Route for serving the prediction form
@app.route('/pred', methods=['GET'])
def form():
    return render_template('predict.html')

# Route for handling form submission and making predictions
@app.route("/pred", methods=['POST'])
def predict():
    # Collect form data
    form_data = {
        'age': float(request.form['age']),
        'TSH': float(request.form['TSH']),
        'TT4': float(request.form['TT4']),
        'T4U': float(request.form['T4U']),
        'FTI': float(request.form['FTI']),
        'T3': float(request.form['T3']),  # Add 'T3' to form data collection
        'sex': request.form['sex'],
        'on_thyroxine': request.form['on_thyroxine'],
        'on_antithyroid_medication': request.form['on_antithyroid_medication'],
        'sick': request.form['sick'],
        'pregnant': request.form['pregnant'],
        'thyroid_surgery': request.form['thyroid_surgery'],
        'I131_treatment': request.form['I131_treatment'],
        'query_on_thyroxine': request.form['query_on_thyroxine'],
        'query_hypothyroid': request.form['query_hypothyroid'],
        'query_hyperthyroid': request.form['query_hyperthyroid'],
        'lithium': request.form['lithium'],
        'goitre': request.form['goitre'],
        'tumor': request.form['tumor'],
        'hypopituitary': request.form['hypopituitary'],
        'psych': request.form['psych'],
        'TSH_measured': request.form['TSH_measured'],
        'T3_measured': request.form['T3_measured'],
        'TT4_measured': request.form['TT4_measured'],
        'T4U_measured': request.form['T4U_measured'],
        'FTI_measured': request.form['FTI_measured'],
    }

    print("Form Data:")
    print(form_data)

  

    # Create DataFrame from form data
    input_data = pd.DataFrame([form_data])

    # Perform one-hot encoding for categorical columns
    input_data_encoded = pd.get_dummies(input_data, columns=columns_to_encode)

    # Ensure that all expected columns are present and in correct order
    input_data_encoded = input_data_encoded.reindex(columns=feature_names, fill_value=0)

    # Check for 'target' column
    if 'target' in input_data_encoded.columns:
        input_data_encoded.drop('target', axis=1, inplace=True)  # Drop 'target' if present

    # Make predictions
    pred = model.predict(input_data_encoded)

    return render_template('predict.html', prediction_text=str(pred[0]))

if __name__ == "__main__":
    app.run(debug=True)
