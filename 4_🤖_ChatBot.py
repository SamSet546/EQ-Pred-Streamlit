import streamlit as st
import numpy as np
from geopy.geocoders import Nominatim

st.title("EarthSense AI")
with st.chat_message("assistant"):
  st.write("Enter your city or state")

if "messages" not in st.session_state:
    st.session_state.messages = []
if location := st.chat_input("Location"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown("Location: "+ location)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": location})
    # calling the Nominatim tool and create Nominatim class
    loc = Nominatim(user_agent="Geopy Library")
    # entering the location name
    getLoc = loc.geocode(location)
    if (getLoc == None):
      with st.chat_message("assistant"):
            st.write("Location not found")
    else:
      with st.chat_message("assistant"):
        st.write(getLoc.address)
        st.write("Latitude = ", getLoc.latitude, "\n")
        st.write("Longitude = ", getLoc.longitude)

else:
    st.write("Please enter a location to proceed.")

#starting to input model to predict
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.spatial import KDTree
import numpy as np

url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTJdjJGMh3LE80MLupe_JRd9u7b6Wv96ZCKyF9P7UBzeMKdgdZpYRicZA9_f33VK0ar9Pnn09wVi-wV/pub?output=csv'
df = pd.read_csv(url)

#Feature Engineering a new variable
tdata=pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQvurwJ539D8fD3PmmSpd5sH3COwvTxqnyDHiwesej06bCDevg8fsuY5dm_lBUxE7xlfn41jKyvHhzQ/pub?output=csv")
coordinates = tdata[['lat', 'lon']].values

# Create the KDTree
tree = KDTree(coordinates)

def calculate_distance_to_boundary(lat, lon):
    # Query KDTree for nearest neighbor (plate boundary)
    dist, idx = tree.query([(lat, lon)])
    return dist[0]

#Classification Model First
df['distance_to_boundary'] = df.apply(lambda row: calculate_distance_to_boundary(row['latitude'], row['longitude']), axis=1)
threshold=5.0

#feature engineering
df['earthquake_occurred'] = (df['magnitude'] >= threshold).astype(int)

x=df[['distance_to_boundary']]
y=df['earthquake_occurred']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


#Inputting the user values
distance_to_boundary = calculate_distance_to_boundary(getLoc.latitude, getLoc.longitude)
st.write("Distance to boundary:", distance_to_boundary)
#Scaling
X_user_normalized = scaler.transform(np.array([[distance_to_boundary]]))
predictions = clf.predict(X_user_normalized)
#outputs
if predictions==0:
  st.write("There is no predicted deadly earthquake (>=5.0) in your area!")
else:
  st.write("Your area may be in danger! There is a predicted deadly earthquake (>=5.0) in your area! Please be cautious!")

#Regression Model
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from scipy import stats
from joblib import Parallel, delayed
import joblib
from sklearn.utils import resample
tdata=pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQvurwJ539D8fD3PmmSpd5sH3COwvTxqnyDHiwesej06bCDevg8fsuY5dm_lBUxE7xlfn41jKyvHhzQ/pub?output=csv")

X= df[['distance_to_boundary']]
y = df[['magnitude']]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)

# Create imputer to fill missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train) # Fit and transform on training data
X_test_imputed = imputer.transform(X_test) # Transform test data using the same imputer

model = MLPRegressor(hidden_layer_sizes=(1,), random_state=1, max_iter=1000)

# Fit the model on the entire training data
model.fit(X_train_imputed, y_train)

#Inputting the user values
distance_to_boundary = calculate_distance_to_boundary(getLoc.latitude, getLoc.longitude)
X_user_normalized = scaler.transform(np.array([[distance_to_boundary]]))
pred2 = model.predict(X_user_normalized)
magnitude = pred2[0]

if predictions==0:
  if magnitude<5.0:
    st.write("If there was an earthquake in your area in the future, the predicted magnitude would be ", magnitude)
  else:
    st.write("If there was an earthquake in your area in the future, the predicted magnitude would be ", magnitude, "This magnitude is above our predicted threshehold because the model isn't perfect!")
else:
  if magnitude<5.0:
    st.write("The magnitude of the dangerous earthquake is predicted to be ", magnitude, "This magnitude is below our predicted threshehold because the model isn't perfect!")
  else:
    st.write("The magnitude of the dangerous earthquake is predicted to be ", magnitude)
from sklearn.tree import DecisionTreeRegressor

def dec_tree_proba(mag):
  X = df[['distance_to_boundary']]
  y = df['magnitude']

  X_scaled = scaler.fit_transform(X)

  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)  #Splitting data

  dec_clf = DecisionTreeRegressor(criterion='squared_error', random_state=100, max_depth=3, min_samples_leaf=5)
  dec_clf.fit(X_train, y_train)

  # Creating a random sample from the entire dataset
  X_sample = X.sample(1, random_state=0)

  # Using the scaler on the sample
  X_sample_scaled = scaler.transform(X_sample)

  mag_proba = dec_clf.predict(X_sample_scaled)
  mag_acc = dec_clf.score(X_test, y_test)

magnitude= magnitude.reshape(1, -1)
st.write(dec_tree_proba(magnitude))
#