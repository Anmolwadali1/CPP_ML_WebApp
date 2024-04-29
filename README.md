# Combined Cycle Power Plant ML Web App

This repository contains a Machine Learning (ML) web application developed using the Streamlit framework. The application is designed to address the problem of predicting the electrical output of a Combined Cycle Power Plant (CCPP) using the provided dataset.

## Functionalities:

**Exploratory Data Analysis (EDA):**
Conduct an in-depth analysis of the dataset to understand its structure, distributions, and correlations among variables. Visualizations such as histograms, scatter plots, and heatmaps are utilized to explore the data.

**Model Training:**
Train various ML models using different algorithms and dataset sizes. This functionality allows users to experiment with different model architectures and dataset subsets to evaluate their performance.

**Predictor:**
Deploy trained ML models to predict the electrical output of the CCPP. Users can input values for ambient variables (Temperature, Ambient Pressure, Relative Humidity, and Exhaust Vacuum) to obtain predictions from the models.

**Inbuilt Tableau Integration:**
Incorporate Tableau for advanced data visualization within the web application. This feature enhances the visualization capabilities, allowing for interactive exploration of the dataset and model predictions.

### Usage:
Clone the repository to your local machine.

Install the necessary dependencies by running pip install -r requirements.txt.

Launch the Streamlit web application using **streamlit run app.py**.

Explore the different functionalities provided by the web application through the user interface.

### Links:
Web App: [Link to Web App](https://mlprojectwebapp.streamlit.app/)


### Dataset:
The dataset used in this project contains 9568 data points collected from a CCPP over six years (2006-2011). It includes hourly average readings of ambient variables such as Temperature, Ambient Pressure, Relative Humidity, Exhaust Vacuum, and the corresponding net hourly electrical energy output.

### Repository Structure:
**app.py**: Main script containing the Streamlit web application.

**requirements.txt**: List of dependencies required to run the application.

**Models/**: Directory for storing trained ML models.

**README.md:** Readme file providing an overview of the project and instructions for usage.

### References:
Combined Cycle Power Plant Dataset from UCI Machine Learning Repository : https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant

### Contributors:
**Anmoljeet Singh Wadali**

**Simranjit Singh Hundal**

**Shabir Ahmed Wani**

Feel free to contribute, provide feedback, or report any issues by opening an issue or submitting a pull request.
