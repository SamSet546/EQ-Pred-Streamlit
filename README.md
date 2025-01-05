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


