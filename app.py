
# Core Pkgs
import streamlit as st 
import os
from PIL import Image 

import time
# EDA Pkgs
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pickle
from ucimlrepo import fetch_ucirepo 

folder_path='./Models'

#Change Page Name & Icon
st.set_page_config(page_title='ML Project',page_icon=':smiley:')


Choose_Model = {"LinearRegression":LinearRegression(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor()
                }


def file_selector(folder_path='./Models'):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	filenames=os.listdir(folder_path)
	print(filenames)
	selected_filename = st.selectbox('Select Model File for Prediction',filenames)
	if selected_filename==None:
		return None
	return os.path.join(folder_path,selected_filename)

def train_model(data,model,test_size=0.2,):
	X = data[['AT','V','AP','RH']]
	y = data['PE']
	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=0)
	model.fit(X_train,y_train)
	print("R2 Train score: {:.2f}".format(model.score(X_train, y_train)))
	print("R2 Test score: {:.2f}".format(model.score(X_test, y_test)))

	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	pickle.dump(model, open(f'{folder_path}/model_{model}.pkl','wb'))
	return (round(model.score(X_train, y_train),2),round(model.score(X_test, y_test),2))

def predict_output(inputs,filename):
	model = pickle.load(open(filename,'rb'))
	pred=model.predict([inputs])
	return round(pred[0],2)

#Helper Functions


# To Improve speed and cache data
@st.cache_data(persist=True)
def get_data():
	from ucimlrepo import fetch_ucirepo 
	combined_cycle_power_plant = fetch_ucirepo(id=294) 
	X = combined_cycle_power_plant.data.features 
	y = combined_cycle_power_plant.data.targets 
	df = X.join(y)
	return df

def main():

	st.title("Welcome to ML Project WebApp")

	st.sidebar.title("ML WebApp")
	menu = ["About","EDA",'Model_Training',"Predictor","Project Demo Video"]
	choice = st.sidebar.selectbox("Choose the following options",menu)
	data = get_data()


	if choice == "About":

		st.subheader("About App")
		    # Popover
		with st.popover("Contributors"):
			st.markdown("Team Contributors\n\n - Anmoljeet Singh Wadali\n\n- Simranjit Singh Hundal\n- Shabir Wani")

		    # Expander
		with st.expander("Click to learn more about WebApp & dataset"):
			html_code ='''<div> 
			<p style="font-size:20px">This is Machine Learning based Webapp for Analysing & predicting the overall hourly Electrical energy output of Combined Cycle Power Plant based on the real-time readings from sensors.</p>
		    </div> 
		    <div><p style="font-size:20px">Features consist of hourly average ambient variables </p>
		   <ul>
           <li>Temperature (T) in the range 1.81°C and 37.11°C</li>
           <li>Ambient Pressure (AP) in the range 992.89-1033.30 milibar</li>
           <li>Relative Humidity (RH) in the range 25.56% to 100.16%</li>
           <li>Exhaust Vacuum (V) in teh range 25.36-81.56 cm Hg</li>
           <li>Net hourly electrical energy output (EP) 420.26-495.76 MW</li>
           </ul>
          The averages are taken from various sensors located around the plant that record the ambient variables every second. The variables are given without normalization. 
        </div>'''
			st.markdown(html_code,unsafe_allow_html=True)
			st.markdown("[Dataset](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)")
			st.markdown('[Notebook](https://colab.research.google.com/drive/1TC3I7Q4bOx0DOioKSgyP47K4deM7yR91?usp=sharing)')

	if choice =='EDA':
		# Title and Subheader
		st.subheader("Exploratory Data Analysis of Dataset")
		if st.checkbox("Preview DataFrame"):
			
			if st.button("Head"):
				st.write(data.head())
			if st.button("Tail"):
				st.write(data.tail())
			else:
				st.write(data.head(2))

		if st.checkbox("Show All DataFrame"):
			
			st.dataframe(data)

		# Show Description
		if st.checkbox("Show All Column Name"):
		
			st.text("Columns:")
			st.write(data.columns)

		# Dimensions
		data_dim = st.radio('What Dimension Do You Want to Show',('Rows','Columns'))
		if data_dim == 'Rows':
			data = get_data()
			st.text("Showing Length of Rows")
			st.write(len(data))
		if data_dim == 'Columns':
			st.text("Showing Length of Columns")
			st.write(data.shape[1])

		if st.checkbox("Show Summary of Dataset"):
			st.write(data.describe().transpose())

		if st.checkbox("Select Columns To Show"):
			df=get_data()
			all_columns = df.columns.tolist()
			selected_columns = st.multiselect('Select',all_columns)
			new_df = df[selected_columns]
			st.dataframe(new_df)

		if st.button("Data Types"):
			st.write(data.dtypes)

		if st.button("Value Counts"):
			st.text("Value Counts By Target/Class")
			st.write(data.iloc[:,-1].value_counts())

		st.subheader("Data Visualization")
		if st.button("Show Correlation Matrix"):
			st.set_option('deprecation.showPyplotGlobalUse', False)
			c_plot=sns.heatmap(data.corr(),annot=True, cmap='coolwarm', fmt=".2f")
			st.write(c_plot)
			st.pyplot()
			# Correlation Matrix


	

	elif choice =='Model_Training':
		st.subheader("Model Training")
		Model = st.selectbox("Choose Model for Training",Choose_Model.keys())
		Test_size = st.number_input("Choose Test size in %",10,100,)

				

		if st.button("Model train"):
			with st.spinner("Training Model...."):
				train_score,test_score = train_model(data,Choose_Model[Model],Test_size/100)
				time.sleep(3)
				st.success(f"Model Trained.\n \n R2 Train score: {train_score} \n \n R2 Test score: {test_score}")

	elif choice =='Predictor':
		st.subheader(f"Prediction of Net hourly Electrical energy for Combined Cycle Power Plant")
		st.text("Choose the input data for Prediction")
		Input_Temperature =  st.slider("Ambient Temperature in Celsius", 1.81,37.11)
		Input_Pressure =  st.slider("Ambient Pressure (AP) in milibar", 992.89,1033.30)
		Input_Humidity =  st.number_input("Relative Humidity (RH) in %", 25.56, 100.16)
		Input_Vaccum =  st.number_input("Exhaust Vacuum (V) in cm Hg", 25.36,81.56)
		X_input = [Input_Temperature,Input_Vaccum,Input_Pressure,Input_Humidity]
		Input_Chosen = f'The Input Chosen : {Input_Temperature} Deg Celsius,{Input_Vaccum} cm Hg,{Input_Pressure} milibar ,{Input_Humidity}%'

		if st.checkbox("<--Show Input Chosen-->"):
			print(X_input)
			st.info(Input_Chosen)

		with st.expander("Show Chosen Input for Prediction"):
			st.info(Input_Chosen)


		filename = file_selector()

		file_selected =st.empty()
		if filename == None:
			file_selected.warning("No Model file Available for Prediction..")
		else:
			file_selected.info(f'Selected Model file:{filename}')

		predicted_result = st.empty()
		if st.button("Predict Output"):
				#output = predict(model,X_input)
				with st.spinner("Predicting the output ...."):
					output = predict_output(X_input,filename)
					#time.sleep(2)
					predicted_result.success(f"Predicted Electrical energy is {output} MW")
		else:
			predicted_result = st.empty()

		# Divider
		st.markdown("""---""")
		st.divider()

	elif choice == "Project Demo Video":
		st.subheader("ML Project Demo Video")
		st.video("https://drive.google.com/file/d/1rs3o2N68b32rUxkx90xpXb_0SgxXkuKC/view?usp=sharing")
		st.markdown("[Project Demo Video](https://drive.google.com/file/d/1rs3o2N68b32rUxkx90xpXb_0SgxXkuKC/view?usp=sharing)")



if __name__ == '__main__':
	main()