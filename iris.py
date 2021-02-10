from datetime import datetime, timedelta

# Airflow
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

# SQL
import sqlite3

# Data
from sklearn.datasets import load_iris

# Preprocessing and Metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import pandas as pd

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


# CREATE DB
def create_db():
   '''
   Create SQLite data base for Iris dataset
   '''

   conn = sqlite3.connect('airflow.db')

   cursor = conn.cursor()
   try:
      cursor.execute("""
         CREATE TABLE iris (
                  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                  SepalLengthCm FLOAT NOT NULL,
                  SepalWidthCm FLOAT NOT NULL,
                  PetalLengthCm FLOAT NOT NULL,
                  PetalWidthCm FLOAT NOT NULL,
                  Species TEXT NOT NULL,
         );
         """)
      
      conn.commit()

      print('Table created successfully.')

   except: 
      print("Table already exists")
      pass

   conn.close()


# DB CONNECT
def db_connect():
   '''
   Connect to DB
   '''

   conn = sqlite3.connect('airflow.db')
   cursor = conn.cursor()
   return conn, cursor


# READ DB
def db_2_df(limit):
   '''
   Read the data from DB
   '''
   
   conn, cursor = db_connect()
   data = pd.read_sql_query(f"SELECT * FROM iris LIMIT {limit}", conn)

   conn.close()

   return data


# GET DATA FROM SCIKIT-LEARN
def get_data():
   '''
   Get iris dataset from scikit-learn datasets and save to SQLite
   '''

   iris = load_iris()
   conn, cursor = db_connect()
   df = pd.DataFrame(iris.data, columns = ["sepal_lenght","sepal_width","petal_lenght","petal_width"])
   df["target"] = iris.target

   try:
      df.to_sql("iris", con=conn, if_exists='replace', index = False)
      print("Successfully saved data on database.")

   except:
      print("Error saving data to SQL")
   
   conn.close()

   


# SPLIT DATASET
def split(data):
   '''
   Train test split
   '''
   x = data.drop("target", axis =1)
   y = data["target"]


   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)

   return x_train, x_test, y_train, y_test


# MODEL COMPETITORS
def competitors():

   models = []

   models.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr')))
   models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
   models.append(('K Nearest Neighbors', KNeighborsClassifier()))
   models.append(('Decision Tree', DecisionTreeClassifier()))
   models.append(('Gaussian NB', GaussianNB()))
   models.append(('Support Vector Machines', SVC(gamma='auto')))
   models.append(('RandomForest', RandomForestClassifier(n_estimators = 50)))
   models.append(('OneR', DummyClassifier()))

   return models


# TRAIN AND EVALUATE
def do_the_job():
   '''
   From a list of classifiers, train and evaluate. 
   '''

   data = db_2_df(limit='150')

   x_train, x_test, y_train, y_test = split(data)

   models = competitors()

   # evaluate each model in turn 
   results = []
   names = []
   for name, model in models:
      kfold = KFold(n_splits=10, random_state=1)
      cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
      results.append(cv_results)
      names.append(name)
      msg = f"{name} - Accuracy: {cv_results.mean()*100}% (+/- {cv_results.std()})\n"
      print(msg)      



# ARGS FOR AIRFLOW
default_args = {
   'owner': 'Jhonkr',
   'depends_on_past': False,
   'start_date': datetime(2021,2,10),
   'email': ['airflow@example.com'],
   'email_on_failure': False,
   'email_on_retry': False,
   'retries': 1,
   'retry_delay': timedelta(minutes=1),
}

# MY DAG
dag = DAG('iris_dag',
         default_args=default_args,
         description='Main DAG for Iris classification problem',
         schedule_interval=timedelta(days=1),
         )

create_db = PythonOperator(
   task_id='sqlite3',
   python_callable=create_db,
   dag=dag)

get_data = PythonOperator(
   task_id='iris_functions',
   python_callable=get_data,
   dag=dag)

training = PythonOperator(
   task_id='machine_learning',
   python_callable=do_the_job,
   dag=dag)

create_db >> get_data >> training