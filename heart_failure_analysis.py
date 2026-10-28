import pandas as pd
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
data.head()
print("Dataset Shape:", data.shape, "\n")

data.columns
data.info()

missing = data.isnull().sum()
print ("\n Total of missing value is:", missing.sum())

features_columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time' ]

target_column = "DEATH_EVENT"

X = data[features_columns]
Y = data[target_column]
counts = Y.value_counts()


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state= 42, stratify= Y, test_size= 0.2 )


import matplotlib.pyplot as plt

print(data.describe(), "\n\n")

counts.plot(kind= "bar")
plt.title("Survival vs Death Distribution")
plt.xlabel("DEATH_EVENT (0=Survived, 1=Died)")
plt.ylabel("Count")
plt.show()
print("Target Distribution:\n", counts, "\n\n\n")


data.hist(figsize= (12, 10))
plt.suptitle("Feature Distributions")
plt.show()
print("\n\n")


corr = data.corr()
plt.figure(figsize=(10, 10))
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation= 90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay as cmd, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


log_model = LogisticRegression(max_iter= 1000, class_weight= "balanced", random_state= 42)
log_pipe = Pipeline([ ("scaler", StandardScaler()), ("model", log_model) ])
log_pipe.fit(X_train, Y_train)

Y_log_prediction = log_pipe.predict(X_test)


print("Logistic Regression:\n")
print("Accuracy: ", accuracy_score(Y_test, Y_log_prediction), "\n")
print("Classification Report: \n")
print(classification_report(Y_test, Y_log_prediction), "\n\n")
print ("Confusion Matrix:\n", confusion_matrix(Y_test, Y_log_prediction), "\n")
cmd.from_predictions(Y_test, Y_log_prediction, display_labels= ["Survived", "Died"])
plt.title("Logistic Regression - Confusion Matrix")
plt.show()
print("\n\n\n")




from sklearn.neighbors import KNeighborsClassifier

KNN_model = KNeighborsClassifier(n_neighbors= 5)
KNN_pipe = Pipeline([ ("scaler", StandardScaler()), ("model", KNN_model)])
KNN_pipe.fit(X_train, Y_train)
Y_KNN_prediction = KNN_pipe.predict(X_test)

print("KNN:\n")
print("Accuracy: ", accuracy_score(Y_test, Y_KNN_prediction), "\n")
print("Classification Report: \n")
print(classification_report(Y_test, Y_KNN_prediction),"\n\n")
print ("Confusion Matrix:\n", confusion_matrix(Y_test, Y_KNN_prediction), "\n")
cmd.from_predictions(Y_test, Y_KNN_prediction, display_labels= ["Survived", "Died"])
plt.title("K-Neighbors Classifier - Confusion Matrix")
plt.show()


from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier ( random_state= 42, max_depth= 4)
tree_pipe = Pipeline([("scaler", "passthrough"), ("model", tree_model)])
tree_pipe.fit(X_train, Y_train)
Y_tree_prediction = tree_pipe.predict(X_test)

print("Decision Tree Classifier:\n")
print("Accuracy: ", accuracy_score(Y_test, Y_tree_prediction), "\n")
print("Classification Report: \n")
print(classification_report(Y_test, Y_tree_prediction),"\n\n\n")
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_tree_prediction), "\n")
cmd.from_predictions(Y_test, Y_tree_prediction, display_labels= ["Survived", "Died"])
plt.title("Decision Tree Classifier - Confusion Matrix")
plt.show()
print("\n\n\n")




from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier (random_state= 42, class_weight= "balanced", max_depth= 6, n_estimators= 200)
RF_pipe = Pipeline([("scaler", "passthrough"), ("model", RF_model)])
RF_pipe.fit(X_train, Y_train)
Y_RF_prediction = RF_pipe.predict(X_test)

print("Random Forest Classifier:\n")
print("Accuracy: ", accuracy_score(Y_test, Y_RF_prediction), "\n")
print("Classification Report: \n")
print(classification_report(Y_test, Y_RF_prediction),"\n\n")
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_RF_prediction), "\n")
cmd.from_predictions(Y_test, Y_RF_prediction, display_labels= ["Survived", "Died"])
plt.title("Random Forest Classifier - Confusion Matrix")
plt.show()


model_comp = pd.DataFrame.from_dict ({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "KNN"],

    "Accuracy": [
        accuracy_score(Y_test, Y_log_prediction),
        accuracy_score(Y_test, Y_tree_prediction ),
        accuracy_score(Y_test, Y_RF_prediction),
        accuracy_score(Y_test, Y_KNN_prediction )
        ],

    "Recall": [
        recall_score(Y_test, Y_log_prediction),
        recall_score(Y_test, Y_tree_prediction),
        recall_score(Y_test, Y_RF_prediction),
        recall_score(Y_test, Y_KNN_prediction)
        ],

    "F1-score": [
        f1_score(Y_test, Y_log_prediction),
        f1_score(Y_test, Y_tree_prediction),
        f1_score(Y_test, Y_RF_prediction),
        f1_score(Y_test, Y_KNN_prediction)
        ]
})
model_comp

from sklearn.model_selection import GridSearchCV

RF_initial = RandomForestClassifier (random_state= 42, class_weight= "balanced")
GS_RF_pipe = Pipeline([ ("scaler", "passthrough"), ("model", RF_initial)])

param_grid = {
    "model__n_estimators": [100, 200, 300 ],
    "model__max_depth": [3, 4, 5, 6, 7, None]
}

grid_search = GridSearchCV (estimator= GS_RF_pipe, param_grid= param_grid, cv= 5, scoring= "f1", n_jobs= -1)
grid_search.fit(X_train, Y_train)

best_RF = grid_search.best_estimator_
Y_best_RF_prediction = best_RF.predict(X_test)

print("Best parameters: ", grid_search.best_params_, "\n")
print("Best CV F1:", grid_search.best_score_,"\n")
print("Tuned Random Forest Accuracy:\n")
print("Accuracy: ", accuracy_score(Y_test, Y_best_RF_prediction), "\n")
print("F1: ", f1_score(Y_test, Y_best_RF_prediction),"\n")
print("Recall: ", recall_score(Y_test, Y_best_RF_prediction),"\n")
print("Classification Report: \n")
print(classification_report(Y_test, Y_best_RF_prediction),"\n\n")
print("Confusion Matrix: ", confusion_matrix(Y_test, Y_best_RF_prediction), "\n")
cmd.from_predictions(Y_test, Y_best_RF_prediction, display_labels= ["Survived", "Died"])
plt.title("Tuned Random Forest - Confusion Matrix")
plt.show()


from sklearn.model_selection import cross_val_score

print("\n Overfitting / Underfitting Check:\n")

train_f1 = f1_score(Y_train, best_RF.predict(X_train))
test_f1 = f1_score(Y_test, Y_best_RF_prediction)

print("Train F1: ", train_f1, "\n")
print("Test F1: ", test_f1, "\n")


cv_scores = cross_val_score(best_RF, X_train, Y_train, cv= 5, scoring= "f1")

print("CV F1 Mean: ", cv_scores.mean(), "\n")
print("CV F1 Std: ", cv_scores.std(), "\n")

from sklearn.ensemble import VotingClassifier

voting_model = VotingClassifier (
    estimators = [
        ("Logistic Regression", log_pipe),
        ("KNN", KNN_pipe),
        ("Decision Tree", tree_pipe),
        ("Random Forest", best_RF) ],

    voting= "hard"
)


voting_model.fit(X_train, Y_train)

Y_voting_prediction = voting_model.predict(X_test)

print("Voting Classifier:\n")
print("Voting Accuracy:", accuracy_score(Y_test, Y_voting_prediction), "\n")
print("F1: ", f1_score(Y_test, Y_voting_prediction),"\n")
print("Recall: ", recall_score(Y_test, Y_voting_prediction),"\n")
print("Classification Report:\n")
print(classification_report(Y_test, Y_voting_prediction), "\n\n")
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_voting_prediction), "\n")
cmd.from_predictions(Y_test, Y_voting_prediction, display_labels= ["Survived", "Died"])
plt.title("Voting Classifier - Confusion Matrix")
plt.show()


model_accuracy_table = pd.DataFrame.from_dict ({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "KNN","Tuned Random Forest", "Voting Classifier" ],

    "Accuracy": [
        accuracy_score(Y_test, Y_log_prediction),
        accuracy_score(Y_test, Y_tree_prediction ),
        accuracy_score(Y_test, Y_RF_prediction),
        accuracy_score(Y_test, Y_KNN_prediction ),
        accuracy_score(Y_test, Y_best_RF_prediction),
        accuracy_score(Y_test, Y_voting_prediction)
        ],

    "Recall": [
        recall_score(Y_test, Y_log_prediction),
        recall_score(Y_test, Y_tree_prediction),
        recall_score(Y_test, Y_RF_prediction),
        recall_score(Y_test, Y_KNN_prediction),
        recall_score(Y_test, Y_best_RF_prediction),
        recall_score(Y_test, Y_voting_prediction)
        ],

    "F1-score": [
        f1_score(Y_test, Y_log_prediction),
        f1_score(Y_test, Y_tree_prediction),
        f1_score(Y_test, Y_RF_prediction),
        f1_score(Y_test, Y_KNN_prediction),
        f1_score(Y_test, Y_best_RF_prediction),
        f1_score(Y_test, Y_voting_prediction)
        ]

})
model_accuracy_table
