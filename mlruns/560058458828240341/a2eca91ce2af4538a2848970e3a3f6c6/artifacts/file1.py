import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # Also required for sns.heatmap

# Load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 5


mlflow.set_experiment("YT-MLOPS-EXP1")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_scores = f1_score(y_test , y_pred ,average='weighted')

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score" , f1_scores)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    
    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

# Save plot
    plt.savefig('Confusion-matrix.png')

# Log artifacts using MLflow
    mlflow.log_artifact('Confusion-matrix.png')
    mlflow.log_artifact(__file__)
    
    
    mlflow.set_tags({"Author" : "sai kumar" , "Project" : "win_classifcation"})
    mloflow.sklearn.load_model(rf , "Random-Forest-Model")
    print(accuracy)
 