from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.utils import load_img, img_to_array

app = Flask(__name__)
picfolder =os.path.join('static','pics')

app.config['UPLOAD_FOLDER'] = picfolder


imagefolder =os.path.join('static','uploads')
app.config['UPLOAD_FOLDER'] = imagefolder
predicted_animal = ''

labels = ['African_Elephant',
          'Amur_Leopard',
          'Arctic_Fox',
          'Chimpanzee',
          'Jaguars',
          'Lion',
          'Orangutan',
          'Panda',
          'Panthers',
          'Rhino',
          'cheetahs']

def predict_animal(img_path=''):
    # dimensions of our images
    img_width, img_height = 256, 256

    # load the model we saved
    model2 = load_model('models\model1.h5')
    model2.compile(loss='sparse_categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

    # predicting images
    img = load_img(img_path, target_size=(img_width, img_height))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model2.predict(x, batch_size=32)
    index = np.argmax(classes, axis=1)[0]
    return labels[index]

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the file from the POST request
    f = request.files['file']
    # Save the file to the uploads folder
    filename = secure_filename(f.filename)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # Call the predict_animal function and pass in the file path
    predicted_animal = predict_animal(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # Render the results template and pass in the predicted animal
    return render_template('results.html', animal=predicted_animal)

@app.route('/results')
def results():
    # Get the predicted animal from the query parameters
    animal = request.args.get('animal')
    # Render the results template and pass in the predicted animal
    return render_template('results.html', animal=animal)


# Load temperature data as a pandas dataframe
# df = pd.read_csv('pca.csv')
# YEAR = df['YEAR'].values
# TEMPERATURE = df['TEMPERATURE'].values

# Fit a linear regression model to the data
# regressor = LinearRegression()
# regressor.fit(YEAR.reshape(-1, 1), TEMPERATURE)

# # Save the trained model to a file
# with open('model.pkl', 'wb') as file:
#     pickle.dump(regressor, file)

# Load the saved model from the file
# predicts temprature and years
with open('model.pkl', 'rb') as file:
    model1 = pickle.load(file)

# for asiatic lion
with open('asiatic_lion.pkl', 'rb') as file:
    model2 = pickle.load(file)

# for leopard
with open('snow_leopard.pkl', 'rb') as file:
    model3 = pickle.load(file)

data = pd.read_csv('ani_info.csv')

@app.route('/')
def index():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'],'img11.jpg')
    pic2 = os.path.join(app.config['UPLOAD_FOLDER'],'don1.jpg')
    pic3 = os.path.join(app.config['UPLOAD_FOLDER'],'don2.jpg')
    pic4 = os.path.join(app.config['UPLOAD_FOLDER'],'don3.jpg')
    pic5 = os.path.join(app.config['UPLOAD_FOLDER'],'don4.jpg')
    pic6 = os.path.join(app.config['UPLOAD_FOLDER'],'don5.png')
    temp1 = os.path.join(app.config['UPLOAD_FOLDER'],'temp1.jpg')
    temp2 = os.path.join(app.config['UPLOAD_FOLDER'],'temp2.jpg')
    temp3 = os.path.join(app.config['UPLOAD_FOLDER'],'temp3.jpg')
    temp4 = os.path.join(app.config['UPLOAD_FOLDER'],'temp4.jpg')
    return render_template('index.html',user_image =pic1,user_image1=pic2,user_image2=pic3,user_image3=pic4,user_image4=pic5,user_image5=pic6,temp_image1 =temp1,temp_image2=temp2,temp_image3=temp3,temp_image4=temp4)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/animal', methods=['POST'])

@app.route('/animal', methods=['POST'])
def animal():
    animal = request.form['animal']
    animal_name = animal.upper()
    if animal_name in data['Name'].values:
        # Extract the row details for the animal
        animal_data = data[data['Name'] == animal_name]

        # Pass the information to the template
        return render_template('animal.html', animal=animal, animal_data=animal_data)
    else:
        return render_template('not_found.html', animal=animal)

@app.route('/', methods=['POST'])
def predict():
     pic1 = os.path.join(app.config['UPLOAD_FOLDER'],'img11.jpg')
     pic2 = os.path.join(app.config['UPLOAD_FOLDER'],'don1.jpg')
     pic3 = os.path.join(app.config['UPLOAD_FOLDER'],'don2.jpg')
     pic4 = os.path.join(app.config['UPLOAD_FOLDER'],'don3.jpg')
     pic5 = os.path.join(app.config['UPLOAD_FOLDER'],'don4.jpg')
     pic6 = os.path.join(app.config['UPLOAD_FOLDER'],'don5.png')
     temp1 = os.path.join(app.config['UPLOAD_FOLDER'],'temp1.jpg')
     temp2 = os.path.join(app.config['UPLOAD_FOLDER'],'temp2.jpg')
     temp3 = os.path.join(app.config['UPLOAD_FOLDER'],'temp3.jpg')
     temp4 = os.path.join(app.config['UPLOAD_FOLDER'],'temp4.jpg')
    
    # Get start_year and end_year values from the form
     start_year = int(request.form['start_year'])
     end_year = int(request.form['end_year'])
     animal = str(request.form['animal_name'])

    # Create an array of future years based on the user's input
     future_years = np.arange(start_year, end_year+1)

    # Predict the temperature for the future years using the linear regression model
     predicted_temperatures = model1.predict(future_years.reshape(-1, 1))

     animal1= animal.lower()
     if animal1=='asiatic lion':
        # prediction model for asiatic lion
        data2 = {'Year': future_years, 'Predicted Temperature': predicted_temperatures}
        df = pd.DataFrame(data2)
        predicted_temperatures_list = df['Predicted Temperature'].tolist()
        future_years_list = df['Year'].tolist()
        future_data = pd.DataFrame({
            'TEMPERATURE':predicted_temperatures_list,
            'YEAR':future_years_list
        })

        result = model2.predict(future_data)
        # result_data = {'Year':future_years,'Predicted Temperature':predicted_temperatures,'Animal Count':result}
        # result_df =pd.DataFrame(result_data)
        # print(result_df.to_string(index=False))

     elif animal1=='snow leopard':
        # prediction model for snow leopard
        data2 = {'Year': future_years, 'Predicted Temperature': predicted_temperatures}
        df = pd.DataFrame(data2)
        predicted_temperatures_list = df['Predicted Temperature'].tolist()
        future_years_list = df['Year'].tolist()
        future_data = pd.DataFrame({
            'TEMPERATURE':predicted_temperatures_list,
            'YEAR':future_years_list
        })
        result = model3.predict(future_data)
        # result_data = {'Year':future_years,'Predicted Temperature':predicted_temperatures,'Animal Count':result}
        # result_df =pd.DataFrame(result_data)
        # print(result_df.to_string(index=False))

     else:
        result='Data not available'
    


    #  animal_name=animal.upper()
    #  if animal_name in data['Name'].values:
    #     # Extract the row details for the animal
    #     animal_data = data[data['Name'] == animal_name]
    #     # Format the information as a string
    #     info_str = f"Name: {animal_data['Common Name'].values[0]}\n"
    #     info_str += f"Taxonomic Group: {animal_data['Taxonomic Group'].values[0]}\n"
    #     info_str += f"Taxonomic Subgroup: {animal_data['Taxonomic Subgroup'].values[0]}\n"
    #     info_str += f"Scientific Name: {animal_data['Scientific Name'].values[0]}\n"
    #     info_str += f"State Conservation Rank: {animal_data['State Conservation Rank'].values[0]}\n"
        
    #     # Print the formatted string
    #     print(info_str)
    #  else:
    #     print(f"{animal} is not present in the animal data")

    # Format the predicted temperatures as a string for display on the webpage
     predicted_temperatures_str = ', '.join([str(round(temp, 2)) for temp in predicted_temperatures])

    # Format the  future years as a string for display on the webpage
     future_years_str = ', '.join([str(round(temp, 2)) for temp in future_years])

    # Render the results page with the predicted temperatures
     return render_template('index.html', predicted_temperatures=predicted_temperatures_str,future_years=future_years_str, result=result,user_image =pic1,user_image1=pic2,user_image2=pic3,user_image3=pic4,user_image4=pic5,user_image5=pic6,temp_image1 =temp1,temp_image2=temp2,temp_image3=temp3,temp_image4=temp4)
    

if __name__ == '__main__':
    app.run(debug=True)



