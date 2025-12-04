
import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi,login

api=HfApi(token=os.getenv('HF_TOKEN'))
DATASET_PATH="hf://datasets/Harsha1001/Tourism-Package-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully")

#Drop the Customer ID column as this is unique for each customer and will not help in prediction.
df=df.drop(columns=['CustomerID'],inplace=True)
df['Gender']=df['Gender'].replace('Fe Male','Female')


label_encoder=LabelEncoder()
df['TypeofContact']=label_encoder.fit_transform(df['TypeofContact'])
df['Occupation']=label_encoder.fit_transform(df['Occupation'])
df['Gender']=label_encoder.fit_transform(df['Gender'])
df['ProductionPitched']=label_encoder.fit_transform(df['ProductionPitched'])
df['MaritalStatus']=label_encoder.fit_transform(df['MaritalStatus'])
df['Designation']=label_encoder.fit_transform(df['Designation'])

x=df.drop(columns=['ProdTaken'])
y=df['ProdTaken']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train.to_csv("xtrain.csv",index=False)
x_test.to_csv("xtest.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
y_test.to_csv("ytest.csv",index=False)

files=["xtrain.csv","ytrain.csv","xtest.csv","ytest.csv"]

for file_path in files:
  api.upload_file(
      path_or_fileobj=file_path,
      path_in_repo=file_path.split("/")[-1],
      repo_id="Harsha1001/Tourism-Package-Prediction",
      repo_type="dataset",
  )

