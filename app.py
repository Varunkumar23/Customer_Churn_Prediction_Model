import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

df_1 = pd.read_csv("tele_churn.csv")

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    input_values = [request.form[f'query{i}'] for i in range(1, 20)]
    
    model = pickle.load(open("trained_model.sav", "rb"))
    model_features = pickle.load(open("model_features.pkl", "rb"))  # Load feature names used during training
    
    data = [input_values]
    
    new_df = pd.DataFrame(data, columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'tenure'
    ])
    
    df_2 = pd.concat([df_1, new_df], ignore_index=True)
    
    # Handle 'tenure' binning
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure'].fillna(0, inplace=True)
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    df_2.drop(columns=['tenure'], axis=1, inplace=True)
    
    # One-hot encode categorical variables
    new_df_dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure_group']])
    
    # Remove duplicate columns if any
    new_df_dummies = new_df_dummies.loc[:, ~new_df_dummies.columns.duplicated()]
    
    # Align feature names with training data
    new_df_dummies = new_df_dummies.reindex(columns=model_features, fill_value=0)
    
    # Make prediction
    single = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1]
    
    if single == 1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)
    
    return render_template('home.html', output1=o1, output2=o2, 
                           **{f'query{i}': request.form[f'query{i}'] for i in range(1, 20)})

if __name__ == "__main__":
    app.run(debug=True)
