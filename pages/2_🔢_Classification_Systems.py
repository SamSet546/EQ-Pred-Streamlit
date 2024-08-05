import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTJdjJGMh3LE80MLupe_JRd9u7b6Wv96ZCKyF9P7UBzeMKdgdZpYRicZA9_f33VK0ar9Pnn09wVi-wV/pub?output=csv'
df = pd.read_csv(url)

st.title('🔢Classification: Predicting the Probability of Magnitude')
st.title('Main Dataset')

#Display the datset 
st.dataframe(df)
st.caption('This data from the Kaggle database, which has data tailored specifically to making predictions with machine learning.')

st.divider()

st.write('For classification, we have the liberty to encode values in our dataset to True or False.')
st.write('In other words, we can assign each value of magnitude to a boolean True/False name.')
st.markdown("""
            - We will add new columns to dataset with titles like magnitude 6.0 
            - If magnitude 6.0 is present in the row of data, the value under the column will be True (vice versa)
            """)

st.divider()

st.write('Notice that the following columns have certain repeating values:')
st.markdown("""
            - type
            - type.1
            - author
            - author.1
            """)

st.write('We can print how many unique values each of these columns have. Look below.')
st.write("'type' column:", df['type'].unique())
st.write("'type.1 column:", df['type.1'].unique())
st.write("'author' column:", df['author'].unique())
st.write("'author.1' column:", df['author.1'].unique())

st.divider()

st.title('Appending New Columns')
st.write('Through a command from the python pandas library, we can make each unique value above its own column.')
st.write('This would allow us to utilize them as independent variables in machine learning classification.')

#Convert 'object' columns into boolean values 
df = pd.get_dummies(df, columns=['type', 'type.1', 'author', 'author.1'])

code = '''
#Convert 'object' columns into boolean values 
pd.get_dummies(df, columns=['type', 'type.1', 'author', 'author.1'])
'''
st.code(code)
st.write('The above code returns this new dataframe.')
st.dataframe(df)
st.caption("A blank box indicates 'False' and a filled box with a checkmark indicates 'True'.")

st.divider()

st.subheader('Establishing the Output Magnitude for Classification')

st.write('Now we must choose a specific value of magnitude to classify.')
st.write('To ensure future success and compatibility with the prediction model, we can use the median magnitude in the dataset:', df['magnitude'].median())
st.write("After calling the pandas command, the 'magnitude 3.7' column now presents itself.")

#Conditioning magnitude values for classification 
df = pd.get_dummies(df, columns=['magnitude'])

st.dataframe(df.loc[:, ['magnitude_3.7']])
st.caption("A blank box indicates 'False' and a filled box with a checkmark indicates 'True'.")
st.write('This column will be the output for our classification prediction model. By the end of this page, we will predict the presence of this magnitude of earthquake based on certain features.')

st.divider()

#Trying multiple input (X) variables 

#Activating classification statsmodel and calculating MLE 

import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

#standardize the X 
from sklearn.preprocessing import StandardScaler

#building the scaler
stscaler = StandardScaler()

st.write('To start, we will return a statistical summary of the following independent variables.')
st.markdown("""
            - Type_ke
            - Type_se
            """)

st.title('Backend of Classification Statistics: Logistic Regression Using StatsModel') 
st.caption('Logistic Regression is a model of regression based on binary inputs/outputs, requiring the values of dependent features to be either 0 (False) or 1 (True).')

X = df[['type_ke', 'type_se']]
y = df['magnitude_3.7']   #Median value of magnitude
 
X_scaled = stscaler.fit_transform(X)

#Adding a sm constant to the X data 
X_scaled = sm.add_constant(X_scaled)

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

#Calling the Log Regress Stats Model 

log_reg_stats_model_1 = sm.Logit(y, X_scaled).fit()
log_reg_stats_model_2 = sm.Logit(y_train, X_train).fit()  #More than 15 iterations, so the model will fail 


st.subheader('With the Standard/Original Data')
st.text(log_reg_stats_model_1.summary())

st.subheader('With the Training Data')
st.text(log_reg_stats_model_2.summary())

st.divider()

st.title('What do these results mean?')
st.write("""
        - The Beta Coefficients are indicated by the 'coef' abbreviation in the lower left corner
            - These values represent the effect of each individual feature on the variance of whether Magnitude 3.7 is True/False
        - As such, the 'type_ke' feature would affect 95.74%/84.36% of the change in the output's boolean value (i.e. True/False)
            - Values under 'type_se' still have a significant influence, at 84.97%/73.94% 
        - The p-values of the features are marked by P > |z| 
            - Generally, if the p-value is less than 0.05, these features would be considered 'statistically significant', or relevant to the output
        - Because the p-values for both features don't satisfy this requirement, it is likely that they are correlated to each other 
            - The standard errors are quite high, and the features come from the same column that we split apart
         """)

st.write("In order to determine whether there is a correlation between the 'type_ke' and 'type_se' features (called multicollinearity), we can use a debugging method called 'Variance Inflation Factor'.")

st.subheader('Variance Inflation Factor')

#Testing for multicollinearity between the 'float' and major 'object' independent variables 
from statsmodels.stats.outliers_influence import variance_inflation_factor 

X = df[['type_ke', 'type_se']]
X = pd.DataFrame(X)

# Converting boolean True/False values to 1 and 0 respectively
bool_columns = X.select_dtypes(include=['bool']).columns
X[bool_columns] = X[bool_columns].astype(int)

# Convert all columns to numeric through coercion
X = X.apply(pd.to_numeric, errors='coerce')

#Assigning these features to a VIF dataframe
vif_data = pd.DataFrame()
vif_data['Features'] = X.columns  #Creating a feature column that will print the name of each feature

#Computing and printing the VIF for each indep variable/feature
vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]

st.write('The Variance Inflation Factor is a measure of how much independent variables affect the variance of each other. It is the standard metric for dealing with multicollinearity within the data.')
st.markdown("""
            - It would be considered the 'R-squared' for the independent features only 
            """)
st.write('Below is a data table with the inflation values. Here is a general rule for interpreting it:')
st.markdown("""
            - A VIF value equal to 1 means the variables are not correlated with each other 
            - A VIF value between 1 and 5 indicates that the variables are semi-correlated
            - A VIF value greater than 5 confirms that variables are highly correlated with each other (that is, multicollinearity)
            """) #source: https://www.investopedia.com/terms/v/variance-inflation-factor.asp#:~:text=In%20general%20terms%2C,variables%20are%20highly%20correlated2

st.dataframe(vif_data)
st.write("Contrary to our suspicion, the 'type_ke' and 'type_se' features are not correlated with each other.")
st.write("Let's view a case where there could a possible autocorrelation/multicollinearity.")

#Testing for multicollinearity between the 'float' and major 'object' independent variables 
from statsmodels.stats.outliers_influence import variance_inflation_factor 

X = df[['latitude', 'longitude', 'depth', 'eventid', 'eventid0', 'type_ke', 'type_se']]
X = pd.DataFrame(X)

# Converting boolean True/False values to 1 and 0 respectively
bool_columns = X.select_dtypes(include=['bool']).columns
X[bool_columns] = X[bool_columns].astype(int)

# Convert all columns to numeric through coercion
X = X.apply(pd.to_numeric, errors='coerce')

#Assigning these features to a VIF dataframe
vif_data = pd.DataFrame()
vif_data['Features'] = X.columns  #Creating a feature column that will print the name of each feature

#Computing and printing the VIF for each indep variable/feature
vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]

st.dataframe(vif_data)
st.write("Notice how when we input multiple features, the stats model senses too many complex connections between the features. Both the 'type_fe' and 'type_se' features have VIF values considerably above 5.")
st.markdown("""
            - This is where overfitting originates, and why we must limit how many independent variables we use. 
            """)

st.divider()

st.subheader('Experimenting with Other Features')

st.write("Recall that another feature that we split was named 'author'. There are numerous unique terms associated with this feature, but we located the most influential two.")
st.write('The new independent variables are as follows:')
st.markdown("""
            - Author_ISC 
            - Author_WEL
            """)

#Testing the author values 

#Activating classification statsmodel and calculating MLE 

X = df[['author_ISC', 'author_WEL']]
y = df['magnitude_3.7']

X_scaled = stscaler.fit_transform(X)

#Adding a sm constant to the X data 
X_scaled = sm.add_constant(X_scaled)

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

log_reg_stats_model_1 = sm.Logit(y, X_scaled).fit()
log_reg_stats_model_2 = sm.Logit(y_train, X_train).fit()

st.subheader('With the Standard/Original Data')
st.text(log_reg_stats_model_1.summary())

st.subheader('With the Training Data')
st.text(log_reg_stats_model_2.summary())

st.markdown("""
            - It is clear that the 'author_ISC' holds greater potential in classifying magnitude, according to the beta coefficients
                - Being the first feature, 'author_ISC' can positively impact the variance in a 3.7 magnitude earthquake by 23.4%
                - As a result, we will add 'author_ISC' to our classification inputs, alongisde 'type_ke' and 'type_se'
            - Also pay attention to the log-likelihood value, which measures how well these parameters would fit into the most probable distribution of parameters (look to GridSearch CV in regression!)
                - For both the standard and trianing data, the 'author' features exhibit a higher log value and thus a better fit to the model than the 'type' features
                - This is further supported by their p-values of 0, satisfying the 0.05 threshold 
            """)

st.divider()

st.write("Finally, we will assess the influence of the 'type.1' features. Unfortunately, the 'author.1' feature does not contain any aliases with statistical signficance to the output.")

st.write('Instead of listing the features, we will display the python script with which the backend code is being run.')

X = df[['type.1_mb', 'type.1_MW', 'type.1_ML', 'type.1_MB']]
y = df['magnitude_3.7']   #Median value of magnitude that we selected 
 
X_scaled = stscaler.fit_transform(X)

#Adding a statsmodel constant to the X data, calibrating the features 
X_scaled = sm.add_constant(X_scaled)

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

code = '''
X = df[['type.1_mb', 'type.1_MW', 'type.1_ML', 'type.1_MB']]  #Independent variables/features
y = df['magnitude_3.7']   #Median value of magnitude that we selected 
 
X_scaled = stscaler.fit_transform(X)

#Adding a statsmodel (sm) constant to the X data, calibrating the features 
X_scaled = sm.add_constant(X_scaled)

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

log_reg_stats_model_1 = sm.Logit(y, X_scaled).fit()  #Building the Standard Data stats model
log_reg_stats_model_2 = sm.Logit(y_train, X_train).fit() #Building the Training Data stats model
'''
st.code(code)
st.caption("In case you're wondering, the 'stscaler' variable is a function that removes the mean of the X data in order to increase the performance of machine learning and stats models.")

log_reg_stats_model_1 = sm.Logit(y, X_scaled).fit()   #Building the Standard Data stats model
log_reg_stats_model_2 = sm.Logit(y_train, X_train).fit()   #Building the Training Data stats model

st.write("Now, let's return the summary of the Logistic Regression stats model we assembled.")

st.subheader('With Standard/Original Data')
st.text(log_reg_stats_model_1.summary())

st.subheader('With Training Data')
st.text(log_reg_stats_model_2.summary())

st.markdown("""
            - From this summary, we can conclude immediately that Feature #2 ('type.1_MW') independently affects the magnitude 3.7 likelihood to the greatest extent
                - Other than Feature #3, this specific sub-feature has a p-value less than 0.05 
            """)

st.divider()

st.header('Official Input Features for Classifying Magnitude')
st.write('Below are the most statistically significant features among all the binary features in the dataset.')
st.markdown("""
            - Type_ke
            - Type_se
            - Author_ISC
            - Type.1_MW
            """)

code = '''
X = df[['author_ISC', 'type_ke', 'type_se', 'type.1_MW']]
y = df['magnitude_3.7']
'''
st.code(code)

from sklearn.preprocessing import OneHotEncoder

X = df[['author_ISC', 'type_ke', 'type_se', 'type.1_MW']]
y = df['magnitude_3.7']

X = X.astype(int)
y_int = y.astype(int)

X_scaled = stscaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_int, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

mod = LogisticRegression()
mod.fit(X_train, y_train)
y_pred = mod.predict(X_test)

#Using StatsModel for the predicted values 

#Computing the residulas of the dependent (regressand) and independent variables (regressors)
residuals = y_test - y_pred

#Sum of Squared Residuals (SSR) 
ssr = np.dot(residuals.T, residuals)

#Sum of squares centered around the mean 
centered_y = y_test - y_test.mean()

#Total Sum of Squares
centered_tss = np.dot(centered_y.T, centered_y)

#Manually calculating the R-squared value with residuals 

st.write('By manually calculating the R-squared number using residuals, or the differences between the actual and predicted magnitude values, we obtain the following:')

R_squared = 1 - (ssr/centered_tss)
st.write('R-squared:', R_squared) #A negative R-squared means the model is poorly fit 
                    #There is no way for the difference of squares to exceed the sum of squares 
st.caption('This tells us that the independent variables above collectively influence 1.6% of the predicted True/False values from a standard Logistic Regression model. The negative sign shows that the Logistic Regression model may be a poor fit for this data.')

st.write('For the rest of this page, only these features will be employed in classification.')
st.markdown("""
            - You will notice very quickly how initializing these independent variables is a pivotal step in increasing the accuracy of our predictions. 
            """)

X = df[['author_ISC', 'type_ke', 'type_se', 'type.1_MW']]
y = df['magnitude_3.7']

X_scaled = stscaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

st.divider()

st.subheader('Correlation Plot / Heat Map')
st.write('Other than Variance Inflation Factor, there is one other way to test whether any of the independent variables are biased to each other.')
st.write("The plot below will identify the correlation between two of the 'type'/'author' features on a scale of 0 to 1.")
st.markdown("""
            - A value of 0 means no correlation is present among the two features
            - A value of 1 means the two features are fully correlated with each other (they are the same feature)
            - Any value between 0 and 1 indicates the degree to which the features are correlated 
            """)

import matplotlib.pyplot as plt
import seaborn as sns

def heatmap(df):
    # Correlation to the target/dependent variable: magnitude
    target_col = 'magnitude_3.7'

    # Convert boolean columns to integers 0 or 1
    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    if target_col not in df.columns:
        raise ValueError(f"The target column '{target_col}' is not numeric or not present in the DataFrame.")
    
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[float, int])
    
    k = 15  # Number of variables to show in the heatmap
    cols = df_numeric.corr().nlargest(k, target_col)[target_col].index
    cm = df_numeric[cols].corr()  # Correlation matrix

    # Plot heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(cm, annot=True, cmap='viridis')
    plt.title('Heatmap of Correlation Matrix')

    # Display the plot in Streamlit
    st.pyplot(plt)

#st.set_option('deprecation.showPyplotGlobalUse', False)  #Disable 'global use' warning 
heatmap(df)

st.write('Because most pairs of features have mutual correlations below 1%, multicollinearity is not an issue and this data will likely not be overfitted. Overfitting occurs when it is difficult to define the relationship between the dependent and independent variables, possibly due to abnormally high variance in the testing features (high multicollinearity).')
st.write('We are ready to move on to the beginning stages of activating the machine learning classification models.')

st.divider()

st.title('Standard Logistic Regression Model')

st.write('Before, we had returned the backend statistics of the Logistic Regression model. Now, we can put it to the test in actually predicting the presence of a Magnitude 3.7 Earthquake, while fitted to the independent variables listed above.')

st.write('The code for training and testing a basic Logistic Regression model is almost the same as that for the Linear Regression model in the previous page! See below.')

st.subheader('The First Predictions')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
    
st.write('The following code will be run in front of you.')

with st.echo():
    # Logistic Regression 
    log_reg_model = LogisticRegression(C=100)  #The higher the C value, the greater the complexity of the model 
                    #The main overfitting risk is avoided as our input variables are not autocorrelated
                              
    #Training the Logistic Regression model on the X and y data                               
    log_reg_model.fit(X_train, y_train)

    #Returning the predictions of the True/False value of a 3.7 Mag Earthquake
    y_pred = log_reg_model.predict(X_test)
    st.dataframe(y_pred)  #Displaying the predictions as a table / dataframe
    
st.write('Notice how nearly all the boxes are blank, thus label the presence of the 3.7 Earthquake as False for each row.')
st.write("While a 3.7 Earthquake isn't much common in the dataset, the model is still not accurate regardless of the score we receive.")

st.header('Accuracy (%)')

acc_1 = log_reg_model.score(X_train, y_train)
acc_2 = log_reg_model.score(X_test, y_test)
st.subheader('Training Accuracy:')
st.write(acc_1*100)
st.subheader('Testing Accuracy:')
st.write(acc_2*100)

st.write('We have achieved 98% accuacy for both the training and testing data.')
st.markdown('''
            - BUT because most of the values in the 'magnitude_3.7' column were False anyways, the model did not experience much difficulty in predicting them
            ''')

st.divider()

st.title('Sampling Methods for the Decision Tree Classifier')

st.write("As with the Decision Tree Regressor, the sklearn python library has a 'tree' class of machine learning models that aims to reduce impurities in the data.")
st.markdown("""
            - All datasets start off as impure: it is a collection of random, scattered data points that need to be linked together 
            - The Decision Tree Classifier classifies magnitude by first fitting data points based on a Sigmoid function like the one shown below 
            """)
st.image('https://ai-master.gitbooks.io/logistic-regression/content/assets/sigmoid_function.png')
st.markdown("""
            - In programming, 0 is False and 1 is True 
                - The upper end of the y-axis is 1.0 and the lower end is 0.0, representing the True/False values of the 3.7 Magnitude column
            - When all the data points are fitted into this function, the Decision Tree will distinctly separate data that fall into a specific range of values and assign them to different classes
            - In the end, each class will contain data points that are similar, meaning the classes are 'pure'
            """)
st.image('https://victorzhou.com/media/random-forest-post/decision-tree2-build2.svg')
st.write('The separation technique can vary, but there are two persistent types: ENTROPY and GINI.')

st.divider()

st.title('Decision Tree: Entropy Criterion')

st.write("'Entropy' is a sampling criterion or method that can be employed as a parameter in the Decision Tree Classifier")

code = '''
DecisionTreeClassifier(criterion='entropy')
'''
st.code(code)

st.write('It functions in the same way as the scientific definition of entropy: randomness.')
st.markdown("""
            - That is, the more disconnected pieces of data present in a certain sample, the greater the entropy
            - When this criterion is active, the Decision Tree Classifer will aim to reduce the value of entropy through as many branches as needed
            - Ideally, the final samples containing the 'pure' data should have an entropy at or near 0
            """)

st.divider()

st.write('The Decision Tree Classifier will intake the following parameters for smoother fitting.')
st.markdown("""
            - random_state = 100
            - max_depth = 3
            - min_samples_leaf = 5
            """)

clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)
st.subheader('Predictions:')
st.dataframe(y_pred)

acc_1 = clf_entropy.score(X_train, y_train)
acc_2 = clf_entropy.score(X_test, y_test)
st.subheader('Training Accuracy:')
st.write(acc_1*100)
st.subheader('Testing Accuracy:')
st.write(acc_2*100)
st.write('The accuracy for both the training and testing data experienced no change.')

st.divider()

st.write('Below is a diagram representing how the Decision Tree prediction model sampled our dataset.')
st.caption('Be sure to pay attention to the value of entropy in each box!')

from sklearn.tree import plot_tree

#Creating a plotting function
def plot_dec_tree(clf, feature_names, class_names):
    plt.figure(figsize=(20, 20))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.show()

fig = plot_dec_tree(clf_entropy, ['X1', 'X2', 'X3', 'X4', 'X5'], ['L', 'B', 'R'])  #High entropy = High impurity
st.pyplot(fig)
st.caption('Consider this diagram similar to a flowchart.')

st.write("At the first stage of separation, the entropy value for the 'False' box is greater than the original data. Only some of the entropy was removed, given that the entropy for the 'True' box is only 0.024 less than the original.")
st.write("At the second stage, there is a significant decrease in entropy in the 'False' path. Judging from the low entropy value of 0.012, the data points within that class must be within the same domain of the classification sigmoid graph, yet only ~1800 out of 35000 samples contain that data.")
st.write("At the final stage, the entropy falls to 0 in the True --> False --> False path and in the False --> False --> True path. An important detail is that apart from some outliers, the classes with the least entropies have the least number of samples. This is a sign of the data being downsampled and cateogrized too precisely, not necessarily to the model's benefit.")

st.divider()

st.title('Decision Tree: Gini Criterion')

st.write("'Gini' is another sampling criterion that is a possible hyperparamater for the Decision Tree Classifier")

code='''
DecisionTreeClassifier(criterion='gini')
'''
st.code(code)
st.write('It crosses paths with the Gini Index, which is a popular metric used to quantify wealth inequality in countries around the world.')
st.markdown("""
            - Like entropy, the 'gini' criterion will progressively create equality between the values in the dataset 
            - Another name for the 'gini index' is impurity, and it specifically returns the proabibility of a random data point being misclassified
            - Thus, as the 'gini' value decreases, classification becomes more accurate 
            """)

st.divider()

st.write('The Decision Tree Classifier will entake this parameter for smoother fitting:')
st.markdown("""
            - random_state = 100
            """)

clf_gini = DecisionTreeClassifier(criterion='gini', random_state=100)  #Gini index measures inequality between values to reduce impurity in the data
clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
print('Predictions:', y_pred)

acc_1 = clf_gini.score(X_train, y_train)
acc_2 = clf_gini.score(X_test, y_test)
st.subheader('Training Accuracy')
st.write(acc_1*100)
st.subheader('Testing Accuracy')
st.write(acc_2*100)  #Accuracy is more than 10% higher than previous variables

st.write('The accuracy for both the training and testing data experienced no change.')

st.divider()

st.write('Below is another similar diagram displaying the process by which Decision Tree sampeld our dataset.')
st.caption('Be sure to pay attention to the value of gini in each box!')

fig = plot_dec_tree(clf_gini, ['X1', 'X2', 'X3', 'X4', 'X5'], ['L', 'B', 'R'])  #High entropy = High impurity
st.pyplot(fig)
st.caption('Consider this diagram similar to a flowchart.')

st.write("While the overall accuracies are the same for both the entropy-driven and gini-driven models, the 'gini' criterion was slightly more thorough in the downsampling process.")
st.write("At the first stage of separation, the 'gini' value only dropped by 0.009 for the 'True' box. This follows the same trend as the 'entropy' model in its first stage.")
st.write("At the second stage, the 'gini' value significantly reduces from 0.037 to 0.002 for 'False' path, but only 5% of the sample was categorized into this low-gini class.")
st.write("At the third stage, the model is quite effective in reducing impurities, converting 60% of the True --> False samples and 75% of the False --> False --> True samples into perfectly classified (0 gini) sets.")
st.write("At the final stage, only one set of samples achieves a 0 'gini' value, while representing only 1% of the its previous size. This stage is quite underwhelming, as only 145 out of around 17000 samples experienced a decrease in the gini index.")

st.write("CONCLUSION: The 'gini' and 'entropy' criterions perform essentially the same way. The percentage splits are almost exactly equal, because Decision Tree tends to split the classes with the lowest entropy value, regardless of the number of samples. All in all, it seems that the Decision Tree model driven by 'gini' contains a marginally higher amount of purified data due to the extra stage of separation.")

st.divider()

st.title('Probability Predictions of ANY Magnitude')

#Text box for user to input magnitude 

mag = st.number_input('Enter a magnitude between 2.5 and 8.3 (only intervals of 0.1 are accepted)')

#Predicting probability with high accuracy 

def dec_tree_proba(mag):
    X = df[['author_ISC', 'type_ke', 'type_se', 'type.1_MW']]
    y = df['magnitude_'+str(mag)]
    
    X_scaled = stscaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)  #Splitting data
    
    dec_clf = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=5)
    dec_clf.fit(X_train, y_train)
    
    # Creating a random sample from the entire dataset
    X_sample = X.sample(1, random_state=0)
    
    # Using the scaler on the sample 
    X_sample_scaled = stscaler.transform(X_sample)
    
    mag_proba = dec_clf.predict_proba(X_sample_scaled)
    mag_acc = dec_clf.score(X_test, y_test)
    
    return mag_proba, mag_acc
    
mag_proba, mag_acc = dec_tree_proba(mag)
dec_tree_proba(mag)

mag_proba_df = pd.DataFrame(mag_proba, columns=['Not Present', 'Present'])

st.write('Probability of Magnitude', mag_proba_df)
st.caption("Notice the small shifts in the 'Not Present' probability when you change the magnitude. When it decreases, it implies that the particular earthquake magnitude inputted is more common.")
st.markdown("""
            - A random sample of the X data was selected, so we are displaying the probability of a certain magnitude when predicted by that sample
            """)
st.write('Accuracy of Prediction', mag_acc)
st.caption('The accuracy for these probabilities fluctates between 96-100% depending on the value of magnitude. This high accuracy may be attributed to the independent variables we initialized in the beginning.')

#st.text_input('Probability of Magnitude:', mag_proba)
#st.text_input('Accuracy of Prediction:', mag_acc)

