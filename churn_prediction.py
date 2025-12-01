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
