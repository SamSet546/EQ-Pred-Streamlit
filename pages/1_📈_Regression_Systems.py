import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTJdjJGMh3LE80MLupe_JRd9u7b6Wv96ZCKyF9P7UBzeMKdgdZpYRicZA9_f33VK0ar9Pnn09wVi-wV/pub?output=csv'
df = pd.read_csv(url)

st.title('📉Regression: Predicting Magnitude')

st.divider()

st.title('Main Dataset')
#Display the datset 
st.dataframe(df)
st.caption('This data from the Kaggle database, which has data tailored specifically to making predictions with machine learning.')

st.divider()

X = df[['longitude', 'latitude']]
y = df['magnitude']

#standardize the X 
from sklearn.preprocessing import StandardScaler

#building the scaler
stscaler = StandardScaler()

X_scaled = stscaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

#Display first 20 rows X and y data 
st.write('X Data (Independent Variables / Explanatory Variables)')
st.dataframe(X.iloc[:20, :])
st.write('y Data (Dependent Variable / Response Variable)')
st.dataframe(y.iloc[:20])

st.divider()

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

df_sub = df.iloc[50000:50500, :]

def scatterplot():
    sns.set_theme()
    plt.figure(figsize=(12, 8))
    plt.title('Latitude vs. Magnitude', size=15)

    # Create a custom colormap excluding the lightest part
    cmap = sns.color_palette("Blues", as_cmap=True)
    custom_cmap = LinearSegmentedColormap.from_list('custom_blues', cmap(np.linspace(0.4, 1, 256)))

    # Plot scatterplot using matplotlib
    plt.scatter(df_sub['latitude'], df_sub['magnitude'], 
                s=np.linspace(60, 600, len(df_sub)), c=df_sub['longitude'], cmap=custom_cmap, marker='o', alpha=0.75)

    # Trendline
    x = df_sub['latitude']
    y = df_sub['magnitude']
    w = np.polyfit(x, y, 1)
    z = np.poly1d(w)
    plt.plot(x, z(x), 'r--')

    plt.colorbar(label='Longitude')
    plt.xlabel('Latitude')
    plt.ylabel('Magnitude')

    # Display the plot in Streamlit
    st.pyplot(plt)

st.title("Visualizing Regression Trends")
#st.set_option('deprecation.showPyplotGlobalUse', False)  #Disable 'global use' warning 
scatterplot()
st.caption('Above is a regression-like scatterplot displaying the relationship between some latitude/longitude and magnitude values from the dataset.')

st.title('What do these results mean?')
st.markdown("""
            - The trend line has an obvious negative slope, where negative latitude values align with the highest and most hazardous earthquake magnitudes
            - The darkest blue colors (highest longitude according to the key) lie at the bottom of the graph, corresponding to the smallest magnitude values
                - Thus, longitude also experiences a negative relationship with the output 
            - The data points are concentrated in the negative latitude range, which represents the Southern Hemisphere of the Earth
            - With respect to the trends for negative latitude and longitude, it is reasonable to assert that the southern and western hemispheres of the Earth experience the most dangerous levels of earthquakes
            """)

st.image('https://study.com/cimages/multimages/16/640px-latitude_and_longitude_of_the_earth_fr.svg8032677541246460831.png')

st.divider()

st.title('Decision Tree Regression Model')
st.write('Decision Tree is a Machine Learning model that makes predictions by splitting data into different classes.')
st.write('The main objective of Decision Tree is to reduce impurities in the data through a specific criterion.')
st.write('The only criterion for regression through Decision Tree is Mean Squared Error (MSE).')
st.markdown('- Note that Mean Squared Error is also a metric we will use to determine the accuracy of our model.')
st.divider()
st.title('Decision Tree in Action')

from sklearn.tree import DecisionTreeRegressor

# Calling the DecisionTreeRegressor from sklearn.tree
dec_tree_regressor = DecisionTreeRegressor(criterion='squared_error', random_state=0)

# Fitting the model to depth first
dec_tree_regressor.fit(X_train, y_train)

# Predicting the values
y_pred_1 = dec_tree_regressor.predict(X_test)

#Printing the code for display on the app
code = '''# Calling the DecisionTreeRegressor from sklearn.tree
dec_tree_regressor = DecisionTreeRegressor(criterion='squared_error', random_state=0)

# Fitting the model to depth first
dec_tree_regressor.fit(X_train, y_train)

# Predicting the values
y_pred_1 = dec_tree_regressor.predict(X_test)'''
st.write('This is the main structure of code used to make predictions:')
st.code(code, language='python')

st.subheader('The Predicted Values vs. Acutal Tested Values (MAGNITUDE)')

#Creating a Predicted / Actual dataframe
pred_df = pd.DataFrame()
pred_df['Predicted Values'] = y_pred_1[:20]
pred_df['Actual Values'] = y_test.iloc[:20].values

#Displaying the dataframe with a Predicted and Actual column
st.dataframe(pred_df)
st.write('You may notice that the predicted values are quite close to the actual values')
st.write('We need to verify these results with the Mean Squared Error accuracy.')
st.write('The Mean Squared Error will tell us the the degree to which the predicted and actual values are not similar.')
st.markdown('- That is: the LOWER the MSE, the BETTER')

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred_1)

acc_1 = dec_tree_regressor.score(X_train, y_train)
acc_2 = dec_tree_regressor.score(X_test, y_test)

st.subheader('Mean Squared Error')
st.text(mse)

st.subheader('Training Accuracy (%)')
st.text(acc_1*100)

st.subheader('Testing Accuracy (%)')
st.text(acc_2*100)

#Transition into debugging with SHAP and bagging regressor!

st.write('As seen above, the Mean Squared Error is quite high and the testing accuracy of the model is negative.')
st.write('Consider that the training accuracy is the only value that is high.')
st.write('This model is overfitted given that the training accuracy is in the best range, but the metrics that depend on our testing data return a low accuracy.')

st.divider()

st.title('Reducing Overfitting and High Variance')
st.write('This model has a high variance, which means we have created a complex fit to the Decision Tree model.')
st.write('Overfitting explains how in a model with high variance, the training accuracy will be high, while the testing accuracy will not.')

st.image('https://media.geeksforgeeks.org/wp-content/cdn-uploads/20190523171258/overfitting_2.png')

st.markdown('Why? This is because the model is too fitted to the values of latitude and longitude we have provided that it fails to predict based on unseen data')

st.divider()
st.header('Model Performance on Unseen Data')

new_df = pd.DataFrame()
new_df['longitude'] = [150]
new_df['latitude'] = [20]

st.dataframe(new_df)
st.caption('A dataframe with random values of latitude and longitude')
st.caption('(These values are very close to the values in the fifth row of the dataset at the top of the page!)')

y_pred_2 = dec_tree_regressor.predict([[150, 20]])

st.subheader('Predicted Magnitude from the Decision Tree Regressor')
st.text(y_pred_2)

st.subheader('Closest Value from the Dataset in Comparison')
st.dataframe(df.loc[5, ['magnitude']])
st.caption('Reminder: This is Row 5 of the original dataset')

#Convert first row of dataset to values only
y_val = df.loc[5, ['magnitude']].values

st.subheader('Mean Squared Error')
mse = mean_squared_error(y_val, y_pred_2)

st.text(mse)

st.write('This final Mean Squared Error justifies an improvement to the model, which we will explore through SHAP, the Bagging Regressor, and GridSearch CV.')

st.divider()

st.title('Debugging: Which features are important?')
st.write('The first step in increasing the accuracy of our regresssion model is only using relevant features as our input (i.e. removing features that have very little effect on the variance of magnitude).')
st.write("For the purpose of the first run, we only assigned 'longitude' and 'latitude' to our X input.")
st.write('We will now consider every numerical feature in the dataset:')
st.markdown("""
            - Depth
            - Longitude
            - Latitude
            - Eventid
            - Eventid0 
            """)

X = df[['depth', 'longitude', 'latitude', 'eventid', 'eventid0']]
y = df['magnitude']

X_scaled = stscaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

dec_tree_reg = DecisionTreeRegressor(random_state=0)

dec_tree_reg.fit(X_train, y_train)

y_pred = dec_tree_reg.predict(X_test)

st.divider()

st.header('SHAP and Error Detection')
st.write('SHAP documentation is rooted in game theory, finding the independent variables that return the most optimal output')

st.write('First we will examine the SHAP bar plot for these variables when entered into the Decision Tree model.')
st.write('In doing so, we must separate the variables into separate arrays.')

#Creating a bar plot using SHAP values 

import numpy as np
from sklearn.linear_model import LinearRegression
import shap

#Making X features independent of each other 

N = 2_000
X_scaled = np.zeros((N, 5))

X_scaled[:1_000, 0] = 1
X_scaled[1_000:1_500, 1] = 1
X_scaled[:250, 2] = 1
X_scaled[500:750, 2] = 1
X_scaled[1_000:1_250, 2] = 1
X_scaled[1_500:1_750, 2] = 1

#mean-center the data
X_scaled[:, 0:3] -= 0.5

y = 2 * X_scaled[:, 0] - 3 * X_scaled[:, 1]

#Checking if variables are independent

np_array = np.cov(X_scaled.T)  #Variables ARE independent of each other 
st.text(np_array)
st.write('Here we are checking that the variables are independent 1D arrays or matrices.')
st.markdown('- Because the output are five separate columns, the separation is successful')

#Checking if variables are mean-centered

mean_cen = X_scaled.mean(axis=0)
st.subheader('Mean Centering')
st.text(mean_cen)
st.write('The SHAP analysis will only function when the variables are mean-centered.')
st.write('Similar to standard deviation, the average of the variables should be adequately distribted with recorded values of the variables.')
st.write('Granted previous SHAP analyses, we can verify centered means because we have 1 value of -0.25 and 4 values of 0.')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

# Build the SHAP explainer
explainer = shap.TreeExplainer(dec_tree_reg)

# Compute SHAP values
shap_val = explainer.shap_values(X_test)

# Display the SHAP bar plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
shap.summary_plot(shap_val, X_test, plot_type='bar') 

st.subheader('SHAP Bar Plot: TreeExplainer')
st.pyplot(fig)
st.write("Feature 4 ('eventid0') is the most impactful to the Decision Tree model, followed by Feature 2 ('latitude') and Feature 1 ('longitude').")
st.write('We can push our SHAP analysis even further with a larger, 64-bit regression model: XG Boost.')

st.divider()

st.subheader('SHAP with XG Boost')

#Creating a better beeswarm plot with XG Boost

import xgboost
xg_mat = xgboost.DMatrix(X_scaled, label=y)
xg_model = xgboost.train({'eta':1, 'max_depth': 3, 'base_score': 0, 'lambda': 0}, xg_mat, 1)

st.subheader('Model Error')
mod_error = np.linalg.norm(y - xg_model.predict(xg_mat))
st.write(mod_error)
st.caption('We created an XG Boost matrix based on the X and y labels - A Model Error of 0 indicates that the predictions made from the matrix do not deviate from the matrix values themself.')

st.subheader('More Stats about XG Boost Parameters')
params = xg_model.get_dump(with_stats=True)[0]
st.text(params)

xg_pred = xg_model.predict(xg_mat, output_margin=True)

explainer = shap.TreeExplainer(xg_model)
explanation = explainer(xg_mat)

shap_val = explanation.values

st.subheader('Absolute Difference')
st.write("The difference between the predicted values produced by the SHAP documentation should be as equivalent to the model's predicted values as possible.")
st.write('The value below is the absolute difference, and should be as close to zero as possible for the SHAP graph to be accurate.')
#should return a value equal to or near 0.0
abs_diff = np.abs(shap_val.sum(axis=1)+explanation.base_values - xg_pred).max()
st.write(abs_diff)

#Displaying the beeswarm plot
fig, ax = plt.subplots()
shap.plots.beeswarm(explanation)
st.subheader('SHAP Beeswarm Plot')
st.pyplot(fig)
st.write("To conclude, for a larger prediction model, the presence of 'depth' and 'longitude' tend to decrease values of magnitude to the greatest degree.")
st.write('If not already clear, the negative region of the plot shows that Feature 1 (longitude) and Feature 0 (depth) have a strong negative relationship with magnitude.')
st.markdown("""
            - That is, increasing values of these features serve to predict smaller values of magnitude
            """)

st.divider()

st.title('Debugging: The Bagging Regressor')
st.write('The Bagging Regressor fits the prediction model to unseen data to improve testing accuracy, taking the Decision Tree Model as an input.')
#Say more

st.subheader('Preparing an Optimized Decision Tree Model with GridSearch CV')

st.write('For top accuracy, SHAP helped us confirm the following features as inputs:')
st.markdown("""
            - Depth 
            - Longitude
            - Eventid0
            """)
st.write('We will use the following outputs')
st.markdown("""
            - Magnitude
            - Eventid
            - Latitude
            """)
st.caption("Note that 'eventid0' and 'longitude' are included to improve model fitting.")

X = df.loc[:, ['depth', 'longitude', 'eventid0']].values
y = df.loc[:, ['magnitude', 'eventid', 'latitude']].values.reshape(-3, 3)  #Adding eventid to the output list

X_scaled = stscaler.fit_transform(X) 

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2021)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
import numpy as np

#Using GridSearch CV for DecisionTreeRegressor and KerasRegressor to optimize parameters 

#DecisionTreeRegressor 

dec_reg = DecisionTreeRegressor()

#Hyperparameters list

st.write('GridSearch CV will accept the Decision Tree Regressor, but requires us to define a list of hyperparameters and certain values associated with them. These hyperparameters change how the prediction model interprets the data.')

st.write("The list used is attached in code below (under the alias: 'search_dict').")

code = '''search_dict = {
    'ccp_alpha': [0.0001, 0.001, 0.01, 0.1], 
    'random_state': [42, 50, 2021, None], 
    'max_depth': [1, 2, 5, 10],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 5, 10, 20]
}
'''
st.code(code)

search_dict = {
    'ccp_alpha': [0.0001, 0.001, 0.01, 0.1], 
    'random_state': [42, 50, 2021, None], 
    'max_depth': [1, 2, 5, 10],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 5, 10, 20]
}

# Calibrating the progress bar and text
st.session_state.progress_bar = st.progress(0)
st.session_state.text_status = "Starting GridSearchCV..."
text_status = st.empty()

from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
import time

#Building a GS CV function that shows the code in progress

import threading
import time

def prog_GS(X, y, progress_bar):
    GS_cv = GridSearchCV(estimator=dec_reg,
                    param_grid=search_dict,
                    scoring=['r2', 'neg_root_mean_squared_error'], 
                    refit='r2', #Will allow us to use sklearn scoring metrics 
                    cv=5,
                    verbose=0) #No verbose makes the progress faster
    
    # Number of total fits
    total_fits = len(search_dict['ccp_alpha']) * len(search_dict['random_state']) * len(search_dict['max_depth']) * len(search_dict['min_samples_split']) * len(search_dict['min_samples_leaf'])
    
    # Define the cross-validation strategy
    cv = KFold(n_splits=5)
    current_fit = 0
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
    for params in ParameterGrid(search_dict):
            dec_reg.set_params(**params)
            dec_reg.fit(X_train, y_train)
            current_fit += 1
            if current_fit % (total_fits // 10) == 0:
                progress = current_fit / total_fits
                progress_bar.progress(progress)
                #st.session_state.text_status = f"Processing fits {current_fit}/{total_fits}"
                text_status.text(f"Processing fits {current_fit}/{total_fits}")
            #time.sleep(0.001)  # Simulate delay
    
    GS_cv.fit(X, y)
    return GS_cv

with st.spinner("Running…"):
        GS_fit = prog_GS(X_train, y_train, st.session_state.progress_bar)
    
# Update session state with the result
st.session_state.GS_fit = GS_fit
st.session_state.computation_done = True

st.divider()

st.subheader('GridSearch CV Fitting Process')

st.write('GridSearch CV will run hundreds of iterations testing the effect of each parameter on the training data and the model.')
st.write('The total number of fits should be visible in the progress bar above.')

# Initialize session state variables
if 'computation_done' not in st.session_state:
    st.session_state.computation_done = False

# Start the background computation if it hasn't been started
if not st.session_state.computation_done:
    threading.Thread(target=run_grid_search).start()

# Run GS CV on the streamlit, displaying results and progress
if st.session_state.computation_done:
    st.text(GS_fit)

    st.write('Below are ALL the best parameters for the Decision Tree model when GridSearch CV is fitted to the training data.')
    st.text(GS_fit.best_estimator_)

    st.write('Below are the best hyperparameters for the Decision Tree model.')
    st.text(GS_fit.best_params_)

st.write('Given these parameters, we will now return the accuracy of the most optimal sklearn.tree model.')
dec_reg = DecisionTreeRegressor(ccp_alpha=0.0001, max_depth=10, min_samples_leaf=20,
                      random_state=42)

dec_reg.fit(X_train, y_train)
y_pred = dec_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

st.write('Predictions: ', y_pred)
st.caption('Only the first column is magnitude! The second column is the predicted eventid and the third column is the predicted latitude values.')
#Return predictions in a data table!

st.write('Training Accuracy: ', dec_reg.score(X_train, y_train))
st.write('Testing Accuracy: ', dec_reg.score(X_test, y_test))
#st.write('Mean Squared Error: ', mse)

st.write('This is a great start for reducing overfitting in the Decision Tree model. Not only have we stabilized our accuracy at 50% (a now positive value), but we have also narrowed the variation between the training and testing accuracy.')
st.write('We can now amplify the success of GridSearch CV with the Bagging Regressor.')

st.divider()

st.header('Initializing the Bagging Regressor')
base_mod_1 = DecisionTreeRegressor(ccp_alpha=0.0001, max_depth=10, min_samples_leaf=20, 
                      random_state=42) #Increasing certain parameters returns greater accruacy 
                                                        #(at the cost of processing speed)
#Comparing DT performance in the Bag Model to Linear Regression
base_mod_2 = LinearRegression(fit_intercept=True)

st.write('The Decision Tree Regressor with the most optimal parameters will be fed into the Bagging Regressor.')
st.write('The basic linear regression model will also be modified by the Bagging Regressor.')
st.write('We will compare the accuracy of the final 2 bagging models, ideally seeing a large difference.')

#Building the Bagging Regressor 
bag_mod_1 = BaggingRegressor(base_mod_1, n_estimators=100, max_features=2)  #increasing the number of estimators = higher accuracy = low mse
                                                  
bag_mod_2 = BaggingRegressor(base_mod_2, n_estimators=100, max_features=2)

code = '''
base_mod_1 = DecisionTreeRegressor(ccp_alpha=0.0001, max_depth=10, min_samples_leaf=20,
                      random_state=42) #Increasing certain parameters returns greater accruacy (at the cost of processing speed)
                                                                                       
#Comparing DT performance in the Bag Model to Linear Regression
base_mod_2 = LinearRegression(fit_intercept=True)

#Building the Bagging Regressor 
bag_mod_1 = BaggingRegressor(base_mod_1, n_estimators=100, max_features=2) 
                                            #increasing the number of estimators = higher accuracy = low mse
bag_mod_2 = BaggingRegressor(base_mod_2, n_estimators=100, max_features=2)
'''

st.code(code)
st.caption('In the reality of the code, the Bagging Regressor takes base_mod_1 (Decision Tree) and base_mod_2 (Linear Regressionn) as parameters.')

#Fitting/training the Bagging Regressor
bag_mod_1.fit(X_train, y_train)
bag_mod_2.fit(X_train, y_train)

#Making predictions based on the testing data (as usual)

y_pred_1 = bag_mod_1.predict(X_test)
y_pred_2 = bag_mod_2.predict(X_test)

st.subheader('Decision Tree Predictions of Magnitude')
st.write(y_pred_1)

st.subheader('Linear Regression Predictions of Magnitude')
st.write(y_pred_2)

st.divider()

#Verify with mean squared error
#mse_1 = mean_squared_error(y_test, y_pred_1)
#mse_2 = mean_squared_error(y_test, y_pred_2)
#st.write('Mean Squared Error DT:', mse_1)  
#st.write('Mean Squared Error LR:', mse_2)
      
#Verify with scoring the accuracy
acc_1 = bag_mod_1.score(X_test, y_test) 
acc_2 = bag_mod_2.score(X_test, y_test)
st.write('Testing Accuracy DT:', acc_1)  
st.write('Testing Accuracy LR:', acc_2)

st.divider()

st.header('Improving Bagging Regressor Results with GridSearch CV')


dec_reg = DecisionTreeRegressor(ccp_alpha=0.0001, max_depth=10, min_samples_leaf=20,
                      random_state=42)

#Base Model = DecisionTreeRegressor 
base_mod = dec_reg

#Building the Bagging Regressor 
bag_mod = BaggingRegressor(base_mod) 

#Hyperparameters list 

search_dict = {
    #'n_estimators': [20, 50, 100], 
    'n_jobs': [10, 15],
    'random_state': [42, None]
}

st.write('Below is the list of hyperparameters that will be calibrated and optimized.')
code = '''
search_dict = {
    #'n_estimators': [20, 50, 100], 
    'n_jobs': [10, 15],
    'random_state': [42, None]
}
'''
st.code(code)

import sys

def prog_GS(X, y):
    st.cache_data.clear()
    
    GS_cv = GridSearchCV(estimator=bag_mod,
                         param_grid=search_dict,
                         scoring=['r2', 'neg_root_mean_squared_error'], 
                         refit='r2', 
                         cv=5,
                         verbose=0)
    
    total_fits = len(search_dict['random_state'])
    current_fit = 0
    cv = KFold(n_splits=5)
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        for params in ParameterGrid(search_dict):
            bag_mod.set_params(**params)
            bag_mod.fit(X_train, y_train)
            current_fit += 1
            if current_fit % (total_fits // 10) == 0:
                st.write(f"Processing fits {current_fit}/{total_fits}")
            # Simulate delay
            time.sleep(0.01)
    
    GS_cv.fit(X, y)
    return GS_cv

st.write('The fitting process will now begin.')

try:
    with st.spinner("Running…"):  
        GS_fit = prog_GS(X_train, y_train)
        st.session_state.GS_fit = GS_fit
        st.session_state.computation_done = True
except BrokenPipeError:
    sys.stderr.write('Broken pipe error occurred.\n')
    sys.stderr.flush()

if st.session_state.computation_done:
    GS_fit = st.session_state.GS_fit
    st.text(GS_fit)

    st.write('Below are ALL the best parameters for the Decision Tree model when GridSearch CV is fitted to the training data.')
    st.text(GS_fit.best_estimator_)

    st.write('Below are the best hyperparameters for the Decision Tree model.')
    st.text(GS_fit.best_params_)

#Returning the final accuracy of the best regression prediction model 
st.divider()

st.write('It is now time to train and test the most complete regression prediction model we can possibly construct.')
st.write('Remember the reasons behind each of the improvements made:')
st.markdown("""
            - SHAP: Find the features in the dataset that significantly affected the variance of the magnitude variable, removing unnecessary features to reduce the complexity of the model fitting.
            - GridSearch CV: Find the hyperparameters of the Decision Tree model that sample the data in the most accurate way possible. 
            - Bagging Regressor: Train the enhanced Decision Tree model on unseen data to improve testing accuracy, which is data that wasn't applied in the model fitting process. 
            """)

dec_reg = DecisionTreeRegressor(ccp_alpha=0.0001, max_depth=10, min_samples_leaf=20,
                      random_state=42)

base_mod = dec_reg

bag_mod_new = BaggingRegressor(base_mod, n_estimators=100, n_jobs=10, verbose=15, max_features=2)  #max_features seems to improve model accuracy

bag_mod_new.fit(X_train, y_train)

y_pred = bag_mod_new.predict(X_test)

st.subheader('Final Predictions and Accuracy')

st.write('Predictions:', y_pred)
st.write('Final Training Accuracy:', bag_mod_new.score(X_train, y_train))
st.write('Final Testing Accuracy:', bag_mod_new.score(X_test, y_test))
#st.write('Final Mean Squared Error', mean_squared_error(y_test, y_pred))

st.divider()
st.write("Now, we will observe the 'official' difference between the magnitude predictions and the test cases.")

st.header('Data Table of Official Predictions vs. Actual Testing Values')
new_df = pd.DataFrame()
new_df['Official Predictions'] = y_pred[0]
new_df['Testing Values'] = y_test[0]

st.dataframe(new_df)

st.divider()

st.title('Final Checks: Cross-Validation')

st.write("An alternate method of returning the accuracy of a model is Cross-Validation. Its function is almost self-explanatory—for every piece of training data that the prediction model fits and then compares with the testing data, a cross-validation model will output a score. This score is generally more accepted than the 'overall accuracy' metric we have been using.")
st.write('Cross-validation is regularly used in neural networks, where data is trained across many different epochs (neurons/nodes). If you would like to learn about how we predicted magnitude with a Convolutional Neural Network (CNN) and Multi-Layer Perceptron (MLP), click the page below.')

#Computing a k-fold cross validation

from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVR

#Assigning the number of folds (how many subsets the data is divided into)

code = '''
num_fold = 5
kf = KFold(n_splits=num_fold, shuffle=True, random_state=42)
'''
st.write("Interestingly, but not surprisingly, KFold is used for both GridSearch CV and Cross-Validation. In fact, the 'CV' in GridSearch CV stands for Cross-Validation.")

st.markdown("""
            - In KFolds, the data as a whole is split into a specific number of folds 
            - Out of those folds, one fold is named the testing data and the rest are the training data 
            - The Cross-Validation model/function will pass through each fold, such that each one will eventually become the testing data
            - Finally, it returns a score based on metrics like mean squared error or a confidence interval 
            """)


num_fold = 5
kf = KFold(n_splits=num_fold, shuffle=True, random_state=42)
st.code(code)
st.write('Here we have created 5 folds, or 5 iterations of model fitting/scoring to run through.')

#Performing the cross-validation

cross_val_results_1 = cross_val_score(bag_mod_new, X_test, y_test, cv=kf)
cross_val_results_2 = cross_val_score(bag_mod, X_test, y_test, cv=kf)

#Printing actual results

st.header('Cross Validation Accuracy (Metric = % Accuracy)')
st.write('With GridSearch CV:', cross_val_results_1)
st.write('Without GridSearch CV:', cross_val_results_2)

st.header('Mean Accuracy across Folds')
st.write('Mean Accuracy With GridSearch CV:', cross_val_results_1.mean())
st.write('Mean Accuracy Without GridSearch CV:', cross_val_results_2.mean())
st.write('While it may seem that Grid Search barely changed the accuracy of the Bagging Regressor model when compared with its standard parameters, it is apparent through KFold Cross-Validation that the enhanced Bagging Regressor with Grid Search performed better across multiple variations of the training and testing data.')

st.subheader('Conclusion')
st.write('Predicting earthquake magnitude is a difficult task, especially considering such factors as location, depth, and the identification linked to the disaster.')
st.write('Yet, in the end, we were able to both construct and debug a machine learning model that could predict earthquake to 50-60% accuracy!')
st.caption("While traditional regression methods may not be the MOST accurate, refer to the 'Classification' page for a better prediction of magnitude.")
