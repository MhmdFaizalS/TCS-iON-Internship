from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib


# load the pre-trained machine learning model
model = joblib.load('Ranking.pkl')
scaler = joblib.load('Scaling_features.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/next')
def next():
    return render_template('next.html')

@app.route('/result', methods=['POST'])
def result():
    # Get user inputs from the form
    battery_power = request.form['battery_power']
    blue = request.form['blue']
    clock_speed = float(request.form['clock_speed'])
    dual_sim = request.form['dual_sim']
    fc = request.form['fc']
    four_g = request.form['four_g']
    int_memory = request.form['int_memory']
    m_dep = float(request.form['m_dep'])
    mobile_wt = request.form['mobile_wt']
    n_cores = request.form['n_cores']
    pc = request.form['pc']
    px_height = request.form['px_height']
    px_width = request.form['px_width']
    ram = request.form['ram']
    sc_h = request.form['sc_h']
    sc_w = request.form['sc_w']
    talk_time = request.form['talk_time']
    three_g = request.form['three_g']
    touch_screen = request.form['touch_screen']
    wifi = request.form['wifi']

    # Scale the user inputs using the pre-trained scaler
    scaled_inputs = scaler.transform([[battery_power, blue, clock_speed, dual_sim, 
                                       fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, 
                                       px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi]])

    # Create a dictionary to hold the user inputs as the model's input
    inputs_dict = {'scaled_inputs': scaled_inputs}

    # Use the model to predict the price range
    predicted_price_range = model.predict(scaled_inputs)
    if predicted_price_range == 0:
        status = 'The Expected Price for this Smartphone is Low '
    elif predicted_price_range ==1:
        status = 'The Expected Price for this Smartphone is Medium'
    elif predicted_price_range == 2:
        status = 'The Expected Price for this Smartphone is High'
    else:
        status = 'The Expected Price for this Smartphone is Very High'
        
    
    
     # Compute the feature importance scores using a machine learning model
    importance_scores = model.feature_importances_

    # Create a DataFrame to store the feature names and their importance scores
    feature_scores = pd.DataFrame({'Feature': ['battery_power', 'blue', 'clock_speed', 'dual_sim', 
                                       'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 
                                       'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time','three_g', 
                                       'touch_screen','wifi'], 'Importance':importance_scores})

    # Sort the features based on their importance scores in descending order
    feature_scores = feature_scores.sort_values(by='Importance', ascending=False)

    # Rank the features based on their importance scores
    feature_scores['Rank'] = np.arange(1, 21)
    
        
                    
        
    # Pass the dictionary to the render_template() function
    return render_template('result.html', result=status,feature_scores=feature_scores.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)
