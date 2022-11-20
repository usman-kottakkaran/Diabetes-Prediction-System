from django.shortcuts import render

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import math


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix,plot_roc_curve, classification_report
from sklearn.metrics import mean_absolute_error , mean_absolute_percentage_error , mean_squared_error , accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


import warnings
warnings.filterwarnings("ignore")


def home(request):
     return render(request, "home.html")
def predict(request):
    return render(request, "predict.html")
def result(request):
    data = pd.read_csv(r"C:\Users\dell\Desktop\Diabet\diabetes_binary_health_indicators_BRFSS2015.csv")
    data["Diabetes_binary"] = data["Diabetes_binary"].astype(int)
    data["HighBP"] = data["HighBP"].astype(int)
    data["HighChol"] = data["HighChol"].astype(int)
    data["CholCheck"] = data["CholCheck"].astype(int)
    data["BMI"] = data["BMI"].astype(int)
    data["Smoker"] = data["Smoker"].astype(int)
    data["Stroke"] = data["Stroke"].astype(int)
    data["HeartDiseaseorAttack"] = data["HeartDiseaseorAttack"].astype(int)
    data["PhysActivity"] = data["PhysActivity"].astype(int)
    data["Fruits"] = data["Fruits"].astype(int)
    data["Veggies"] = data["Veggies"].astype(int)
    data["HvyAlcoholConsump"] = data["HvyAlcoholConsump"].astype(int)
    data["AnyHealthcare"] = data["AnyHealthcare"].astype(int)
    data["NoDocbcCost"] = data["NoDocbcCost"].astype(int)
    data["GenHlth"] = data["GenHlth"].astype(int)
    data["MentHlth"] = data["MentHlth"].astype(int)
    data["PhysHlth"] = data["PhysHlth"].astype(int)
    data["DiffWalk"] = data["DiffWalk"].astype(int)
    data["Sex"] = data["Sex"].astype(int)
    data["Age"] = data["Age"].astype(int)
    data["Education"] = data["Education"].astype(int)
    data["Income"] = data["Income"].astype(int)

    data["Diabetes_binary_str"] = data["Diabetes_binary"].replace({0: "NOn-Diabetic", 1: "Diabetic"})

    colomns = ["Fruits", "Veggies", "Sex", "CholCheck", "AnyHealthcare", "Diabetes_binary_str"]

    data.drop(colomns, axis=1, inplace=True)

    X = data.drop("Diabetes_binary", axis=1)
    Y = data["Diabetes_binary"]

    #from imblearn.under_sampling import NearMiss
    #nm = NearMiss(version=1, n_neighbors=10)
    #x_sm, y_sm = nm.fit_resample(X, Y)

    #X_train, X_test, Y_train, Y_test = train_test_split(x_sm, y_sm, test_size=0.3, random_state=42)

    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler()
    #X_train = scalar.fit_transform(X_train)
    #X_test = scalar.fit_transform(X_test)

    #rf = RandomForestClassifier(max_depth=12, n_estimators=10, random_state=42)
    ​#rf.fit(X_train, Y_train)



    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    val11 = float(request.GET['n11'])
    val13 = float(request.GET['n13'])
    val14 = float(request.GET['n14'])
    val15 = float(request.GET['n15'])
    val16 = float(request.GET['n16'])
    val17 = float(request.GET['n17'])
    val19 = float(request.GET['n19'])
    val20 = float(request.GET['n20'])
    val21 = float(request.GET['n21'])


    #pred = rf.predict([[val1, val2,val4, val5, val6, val7, val8, val11, val13, val14, val15, val16, val17, val19, val20, val21]])


    result1=" "
    #if pred==[1]:
     #   result1 = "diabetic"
    #else:
     #   result1 = "no diabetic"

    return render(request, "predict.html", {"result2":result1})


