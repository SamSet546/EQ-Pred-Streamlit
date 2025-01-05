# Why this app? 

- Provide an application platform for people to experience both the backend and frontend design of machine learning prediction systems
- While this application does not offer a thorough walkthrough of the code used, we explain the problem solving process that leads to making accurate predictions
- This application may mirror the focus of programming websites and articles like GeeksforGeeks or Medium
- For this application, we will showcase an environmental monitoring algorithm trained at predicting the magnitude of earthquakes around the world


I built this app for a technical internship under the company Billion Dollar Startup Ideas. The contents of this app are applicable to all people, non-coders included. I originally intended for this app to be understood by the company CEO (an author, not a coder) in the final presentation I delivered to him, showcasing each feature of the app. It contains multiple versions of an earthquake magnitude prediction algorithm, including regression and classification methods, a spatial map of all the earthquakes from the dataset I predicted from, and a earthquake-safety chatbot created by my internship partner. Feel free to explore on your own by reading the directions below!


# On Visual Studio Code

## Installing Pip (Mac) 
1. Download get-pip.py (https://pip.pypa.io)
2. Install pip: $ python get-pip.py

## Running the application on a local host server 
1. Create a new folder and a new python file (.py) under the folder
2. Make sure that any Python 3.12 and environment packages are installed
3. Copy and paste the code from the 1_🌎_Main_Page.py file in this Github repository into the python file you created
4. In the new terminal, type: 

```
pip install streamlit
```

(streamlit is a Python-friendly application builder that we used to display our machine learning algorithm in an interactive and intuitive way)

4. Create a new folder under the main folder titled "pages"
5. Under the "pages" folder, create four new python (.py) files
6. Copy and paste the code from the files from my pages folder so that each of your files correspond to my files (i.e. 1_📈_Regression_Systems.py, 2_🔢_Classification_Systems.py, 3_🗺️_Geostationary_Mapping.py, 4_🤖_ChatBot.py)
7. In the same terminal, type:
```
source .venv/bin/activate
```
8. Then type:
```
streamlit run 1_🌎_Main_Page.py
```
9. This input should take you to a local host tab on your broswer tab displaying the home page (1_🌎_Main_Page.py) of the website
10. To view a different page, simply click on the page on the left sidebar 


