# COVID-19-Tweet-Predictor

In this Repository I have made a main file wich is the main flask file or the app file.It is app.py.It handles all the request for the server.
The Other important file is model. It is the file in which I have Trained My model for predicting That wether the tweet is related to COVID-19 or not.
It is The main Machine Learning File.

The data on which my model is trained is datasets_test and datasets_train file which is the CSV file.

Now classifier.pkl and cv.pkl are the pickle files which are then read by our app file for predicting the Tweet.

After that the Template Folder has Our HTML files for webpage. It has three files which are index.html, predict.html and base.html. 
The index file is our main home page file.
The predict file is for getting the data. 
The base file is the basic or base holder html file which has the main code for the other two html files.
The two index.html and predict.html afiles are connected to base.html file using jinja template.

Now I have Deployed my web application on cloud using heroku platform which is very good one. 
For Deploying our model on Heroku We need Two more files. 
One is requirements.txt in which we write the reqiuired version of each module or library needed like Python , pandas, numpy etc.
The othe file is Procfile it tells the Heroku server that which file to run first or from which file server will start. 
PLEASE NOTE :--- The Procfile name is case sensitive and also it should not have any extension . 
It means that it should be without any extension.So be careful :)

The Link To The website for my deployed Website is:--- 

If you like . Please Star my Repo :)
