import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df=pd.read_csv("data1.csv")

fraud_row = df[df['Class']==1].iloc[0]
test_input = fraud_row.drop('Class')


X=df.drop('Class',axis=1)
y=df['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)



raw_v_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                   'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
raw_known_features = ['Time', 'Amount']

preprocessor = ColumnTransformer(
    transformers=[
        ('raw_known_features', 'passthrough', raw_known_features),
        ('raw_v_features', 'passthrough', raw_v_features)

    ]
)

pipeline_rf = Pipeline(
    [
        ('preprocessor', preprocessor),
        ('rf_model', RandomForestClassifier(
                    random_state=42,
                    n_estimators=20,
                    n_jobs=-1
         )
        )
    ]
)

pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)
y_probs_rf = pipeline_rf.predict_proba(X_test)[:, 1]
threshold_rf = 0.4
y_pred_rf_custom = (y_probs_rf >= threshold_rf).astype(int)


# print(classification_report(y_test,y_pred_rf_custom))
# print(confusion_matrix(y_test,y_pred_rf_custom)) 

joblib.dump(pipeline_rf, 'random_forest_pipeline.pkl')
print("Model training complete.")
print("Pipeline saved as random_forest_pipeline.pkl")
#BEST THRESHOLD = 0.4
