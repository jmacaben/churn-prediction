import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Churn Modeling.csv')

# DATA EXPLORATION

print(data.tail(n=5))   # Viewing a few rows to see if data loaded correctly
print(data.describe())  # Summary statistics
print(data.dtypes)      # Data types of each column

# Checking unique values in categorical columns
categorical_columns = ['Surname', 'Geography', 'Gender']
for col in categorical_columns:
    print(f"--- {col} ---")
    print(data[col].value_counts())


# Histograms
# Normalized distribution of each numerical variable between churners and non-churners
# From this, we can see whether churners have different score/salary/age patterns, check for skewness or multimodal behavior, or just identify variables that visually separate the two classes
numerical_columns = ['CreditScore','Age','Balance','EstimatedSalary']
data_num = data[numerical_columns].copy()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_num[numerical_columns] = scaler.fit_transform(data_num)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(numerical_columns):
    ax = axes[i]
    ax.hist(data_num[data['Exited'] == 0][col].dropna(), bins=20, color='blue', alpha=0.5, label='Exited=0', density=True)
    ax.hist(data_num[data['Exited'] == 1][col].dropna(), bins=20, color='red', alpha=0.5, label='Exited=1', density=True)
    ax.set_title(f'{col} Distribution by Exited (Normalized)')
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.legend()
plt.tight_layout()
plt.show()
# Observations: Salary has a uniform distribution whereas Balance has a normal distribution plus a heavy outlier (lots of 0 balances)

# Boxplots
# Used to see how the central values differ between churners vs non-churners
# Complement histograms by focusing on central tendency and spread rather than overall shape
for col in ['Age','CreditScore','Balance']:
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Exited', y=col, data=data)
    plt.title(f'Box plot of {col} by Exited')
    plt.show()
# Observations: Churners tend to have a higher median age; perhaps younger people care less and don't want to go through the hassle of switching banks
              # CreditScore shows little difference between churners and non-churners
              # A lot of customers with zero balance stay at the bank, possibly because they have less incentive or fewer opportunities to switch

# Correlation heatmap
corr_cols = ['CreditScore','Age','Tenure','NumOfProducts','HasCrCard','Balance','EstimatedSalary','Exited','IsActiveMember']
plt.figure(figsize=[16,8])
sns.heatmap(data[corr_cols].corr(), cmap=sns.diverging_palette(20,220,n=200), annot=True)
plt.title("Correlation between numerical features")
plt.show()
# Observations: Most features in this dataset are weakly correlated

#Scatter plots
numerical_columns = ['Balance', 'EstimatedSalary', 'CreditScore', 'Age', 'Tenure']
pairs = [
    ('Balance', 'EstimatedSalary'),
    ('Balance', 'CreditScore'),
    ('Balance', 'Tenure'),
    ('Age', 'Tenure')
]

for x_col, y_col in pairs:
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=x_col,
        y=y_col,
        hue='Exited',
        data=data,
        palette={0: 'blue', 1: 'red'},
        alpha=0.6
    )
    plt.title(f'{y_col} vs {x_col} by Exited')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title='Exited')
    plt.show()
# Observations: Across these pairs of variables, churners and non-churners overlap heavily

#Two-way table
def plot_two_way(data, group_col, desc=None):
    table = pd.crosstab(
        data[group_col],
        data['Exited'],
        margins=True,        
        margins_name='Total'
    )

    label = desc if desc else group_col

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        table,
        annot=True,
        fmt='g',
        cmap='Blues'
    )
    plt.title(f'{label} vs Exited — Two-Way Table')
    plt.xlabel("Exited")
    plt.ylabel(group_col)
    plt.show()

plot_two_way(data, 'Gender')
plot_two_way(data, 'Tenure')
plot_two_way(data, 'Geography')
# Observations: Customers from Germany exhibit significantly higher churn rates compared to France and Spain.
              # Tenure does not show a strong linear relationship with churn, but some mid-range tenure groups appear slightly more likely to leave


# DATA PREPARATION AND FEATURE ENGINEERING

df = data.copy()

# Dropping irrelevant features
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

# Converts numeric credit scores into categorical buckets (Fair, Good, Very Good, Excellent)
bins = [-float('inf'), 559, 659, 724, 759, float('inf')]
labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']

df['CreditScoreBucket'] = pd.cut(df['CreditScore'], bins=bins, labels=labels, right=True, include_lowest=True)

# Combining pairs of features to make new ones
df['Balance_to_Salary'] = df['Balance']/df['EstimatedSalary']
df['Age_Tenure'] = df['Age']*df['Tenure']
df['Product_to_Tenure'] = df['NumOfProducts']/(df['Tenure']+1)

# Combining multiple features to calculate an 'Engagement Score'
from scipy.optimize import minimize

def engagement_score(df, w):
    # Defining Engagement Score as a linear combination of IsActiveMember, HasCrCard, NumOfProducts, and Tenure
    # w = [w1, w2, w3, w4]
    return w[0]*df['IsActiveMember'] + w[1]*df['HasCrCard'] + w[2]*df['NumOfProducts'].astype(int) + w[3]*df['Tenure']

def correlation_loss(w, df, target):
    score = engagement_score(df, w)
    corr = np.corrcoef(score, target)[0,1]
    return 1 - abs(corr)  # Minimize 1 - |corr| to maximize absolute correlation

w0 = np.ones(4)  # Starting with equal weights
res = minimize(correlation_loss, w0, args=(df, df['Exited']))
print(res.x)

w = res.x # Gives optimized weights

# Using the optimized weights, we calculate the final Engagement Score for each customer
df['EngagementScore'] = w[0]*df['IsActiveMember'] + w[1]*df['HasCrCard'] + w[2]*df['NumOfProducts'].astype(int) + w[3]*df['Tenure']

# One-hot encoding
df['Gender'] = df['Gender'].map({'Male':1,'Female':0})

def oneHotEncoding(df, col:str):
    return pd.get_dummies(df, columns=[col])

df = oneHotEncoding(df, 'Geography')
df = oneHotEncoding(df, 'CreditScoreBucket')

# Another heatmap for visualizing the new columns
columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited', 'EngagementScore', 'Balance_to_Salary', 'Age_Tenure', 'Product_to_Tenure', 'Geography_France', 'Geography_Germany', 'Geography_Spain', 'CreditScoreBucket_Poor', 'CreditScoreBucket_Fair', 'CreditScoreBucket_Good', 'CreditScoreBucket_Very Good', 'CreditScoreBucket_Excellent']

correlations = df[columns].corr()
plt.figure(figsize=[32,16])
plt.title("Correlation between numerical features")
sns.heatmap(correlations, cmap=sns.diverging_palette(20, 220, n=200), annot= True)
plt.show()

# Observations: Poor correlation across the entire dataset for numerical columns

# MODEL TRAINING AND EVALUATION

# Classification
# Trying XGBoost, Random Forest, Logistic Regression, and K-Nearest Neighbours
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

# Separate features and target
X = df.drop(columns=['Exited'])
y = df['Exited']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

results = []

for name, model in models.items():
    # Use scaled features for logistic regression + KNN
    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append([name, acc, prec, rec, f1])

# Display metrics
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\n=== Classification Model Performance ===\n")
print(results_df.to_string(index=False))

# Observations: Among the tested models, Random Forest and XGBoost perform the best overall, with the highest accuracy and F1 scores
              # Likely because they handle non-linear interactions and categorical variables better


# Regression
# Predicting tenure for churned customers using
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Filter for churners only
df_reg = df[df['Exited'] == 1].copy()

# Separate features and target
# Remove leakage columns, i.e. columns that have Tenure directly used in their calculations
X_reg = df_reg.drop(columns=['Tenure', 'Exited', 'EngagementScore'])
leak_cols = [c for c in X_reg.columns if "Tenure" in c]
X_reg = X_reg.drop(columns=leak_cols)

y_reg = df_reg['Tenure']

# Train-test split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_r_scaled = scaler.fit_transform(X_train_r)
X_test_r_scaled = scaler.transform(X_test_r)

models_r = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.001),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42
    ),
    "XGBoost Regressor": XGBRegressor(
        n_estimators=300,
        learning_rate=0.07,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )
}

results_r = []

for name, model in models_r.items():
    if name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
        model.fit(X_train_r_scaled, y_train_r)
        preds = model.predict(X_test_r_scaled)
    else:
        model.fit(X_train_r, y_train_r)
        preds = model.predict(X_test_r)

    # Metrics
    mae = mean_absolute_error(y_test_r, preds)
    mse = mean_squared_error(y_test_r, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_r, preds)

    results_r.append([name, mae, mse, rmse, r2])

# Display results
results_df_r = pd.DataFrame(
    results_r,
    columns=["Model", "MAE", "MSE", "RMSE", "R2 Score"]
)

print("\n=== Regression Model Performance (Predicting Tenure of Churners) ===\n")
print(results_df_r.to_string(index=False))

# Observations: All regression models so very poor performance
              # Predicting tenure before a customer exits is challenging with only demographic and basic account information
              # Thus, more detailed behavioral or transactional data would likely be needed for meaningful predictions