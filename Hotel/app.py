
from flask import Flask,render_template,request
import pickle
import numpy as np 
#Define
flask_app_=Flask(__name__,template_folder='templates')
input_name=['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
       'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
       'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month',
       'arrival_date', 'market_segment_type', 'repeated_guest',
       'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
       'avg_price_per_room', 'no_of_special_requests']
cat_features=['type_of_meal_plan','room_type_reserved','market_segment_type']

model = pickle.load(open(r'NoteBooks\\model.pkl','rb'))
        
@flask_app_.route('/')
def home():
    return render_template('home.html')

@flask_app_.route('/predict',methods=['POST'])
def predict():
    features=[]
    for col in input_name:
        value=request.form.get(col)
        if col in cat_features:
            le=pickle.load(open(r'NoteBooks\{}_le.pkl'.format(col),'rb'))
            v=le.transform(np.array([[value]]))
            features.append(v)
        else:
            features.append(float(value))
    x=np.array(features).reshape(1,17)
    y_pred=model.predict(x)
    # predict_proba
    if y_pred==1:
        output='the customer may confirm reservation'
    else:
        output='the customer may cancel reservation'
        
    return render_template('result.html',prediction_test=output)

if __name__=='__main__':
    flask_app_.run()


