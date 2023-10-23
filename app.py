#------------------------------------------------- Libraries ------------------------------------------------- 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


#------------------------------------------------- Read Dataset ------------------------------------------------- 

data = pd.read_csv("datasets_9768_13874_HR_comma_sep.csv")


#------------------------------------------------- Info Dataset ------------------------------------------------- 
print(data.head()) 
data.info() 
data.isnull().sum()  
print("Formato do dataset:", data.shape) # result (14999, 19)
data.columns
#------------------------------------------------- Data Cleaning ------------------------------------------------- 

# rename column name from "sales" to "departament"
data = data.rename(columns= {"sales":"department"})
print("Departamentos únicos:", data["department"].unique())

#------------------------------------------------- Data Analysis ------------------------------------------------- 

#The left column is the outcome variable recoring 1 and o.1 for employees who left the company\
#and 0 for those who are still working here.
# The department column of the dataset has many categories an we need to reduce the categories \
# for better modelling. Let's see all the categories of the department column.
# Letzs add all the 'technical', 'support and 'TI' columns into one column to make our analysis easier.
data["department"] = np.where(data["department"] == "support", "technical", data["department"])
data["department"] = np.where(data["department"] == "IT", "technical", data["department"])

#------------------------------------------------------------------------------------------------------ 

# Creating Variables for Categorical Varibles
# As there are two categorical variables(deparmente, salary)in the dataset and they need to be\
# converted to dummy varyables before they can be used for modelling.
cat_vars = ["department","salary"]
for var in cat_vars:
    cat_list="var"+"_"+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1

#------------------------------------------------------------------------------------------------------ 
# Now the actual variables nees to be removed after the dummy variable have been created. Column names\
# after creating dummy variables for categorical variables
data.drop(data.columns[[8,9]],axis=1,inplace=True)
print("Nomes das colunas após a criação das variáveis dummy:", data.columns.values)

#------------------------------------------------------------------------------------------------------ 

# The outcome variable is "left", and all the other variables are predictors.
data_vars=data.columns.values.tolist()
y=["left"]
x=[i for i in data_vars if i not in y]

#------------------------------------------------------------------------------------------------------ 

# Calculate percentage of employess who "
porcentagem_saída = (data["left"].sum() / data.shape[0])*100 #result = 23.80825388359224

#------------------------------------------------- Model Predict ------------------------------------------------- 

cols=['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years','department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low'] 
X=data[cols]
y=data['left']

# Splitting data into train an test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Creating and training logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#  Prediction on test set
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# View results
print("Acurácia do modelo: {:.2f}%".format(accuracy * 100))
print("Matriz de confusão")
print(confusion)
print("\nRelatório de Classificação")
print(report)

#------------------------------------------------------------------------------------------------------ 

# The performance of our model is not good because the data set is unbalanced it by generating synthetic data
cols=['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years','department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low'] 
X=data[cols]
y=data['left']

# Splitting data into train an test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(sampling_strategy="auto")
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#  Creating and training logistic regression model with rebalanced data

# Predict on test set
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# View results
print("Acurácia do modelo com dados de teste: {:.2f}%".format(accuracy * 100))
print("Matriz de Confusão:")
print(confusion)
print("Relatório de Classificação:")
print(report)

# Our Model improved in detecting class 1(employees leaving) compared to the unbalanced model.
# The recall for classa 1 also increased to 78% after balancing the classes while in the previous one it was 0.24

#------------------------------------------------------------------------------------------------------ 
# Now we will adjust the hyperparameters to try to improve the performance of the algorithm

# Hyperparameter grid for search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 200, 300]
}

# Create a goal to find the best hyperparameters
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="accuracy" )

# Train the model with the best combination of hyperparameters
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the models with the best hyperparameters
best_model = grid_search.best_estimator_

# Predict Model
y_pred= best_model.predict(X_test)

# Model evaluation with test data
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# View results
print("Acurácia do modelo com dados de teste: {:.2f}%".format(accuracy * 100))
print("Matriz de Confusão:")
print(confusion)
print("Relatório de Classificação:")
print(report)

# The recall for claa 1 ( employeed leaving)improved even futher, reaching 80%. This means the model is better identifying employees\
# Who are likely to leave. Accuracy


#------------------------------------------------------------------------------------------------------ 

# Rando forest
cols=['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years','department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low'] 
X=data[cols]
y=data['left']

# Divisão dos dados em conjunto de train e test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar um modelo
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=42)

# Ajuste do modelo
rf_model.fit(X_train, y_train)

# Previsão com o modelo de teste
y_pred = rf_model.predict(X_test)

# Avaliação do modelo com os dados de teste
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Exibir os resultados
print("Acurácia do modelo com dados de teste: {:.2f}%".format(accuracy * 100))
print("Matriz de Confusão:")
print(confusion)
print("Relatório de Classificação:")
print(report)


#------------------------------------------------------------------------------------------------------ 
# Obter a importância das variáveis a partir do modelo Random Forest
feature_importance = rf_model.feature_importances_

# Normalizar as importâncias para que a soma seja igual a 100%
total_importance = sum(feature_importance)
normalized_importance = [imp / total_importance * 100 for imp in feature_importance]

# Criar um dicionário que mapeia o nome do departamento à sua importância normalizada
department_importance = dict(zip(cols, normalized_importance))

# Classificar os departamentos com base em sua importância
sorted_departments = sorted(department_importance, key=department_importance.get, reverse=True)

# Exibir os departamentos mais influentes na rotatividade em porcentagem
print("Departamentos mais influentes na rotatividade (em porcentagem):")
for department in sorted_departments:
    print(f"{department}: {department_importance[department]:.2f}%")












