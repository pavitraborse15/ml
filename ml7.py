
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Dataset
data = [
    ['Milk', 'Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter'],
    ['Bread'],
    ['Milk', 'Bread', 'Butter'],
    ['Milk', 'Bread']
]

# Convert data
te = TransactionEncoder()
df = pd.DataFrame(te.fit_transform(data), columns=te.columns_)

# Frequent itemsets
items = apriori(df, min_support=0.3, use_colnames=True)
print("Frequent Itemsets:\n", items)

# Association rules
rules = association_rules(items, metric="confidence", min_threshold=0.6)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Find outliers using transaction length
length = df.sum(axis=1)
Q1, Q3 = length.quantile(0.25), length.quantile(0.75)
IQR = Q3 - Q1
outliers = df[(length < Q1 - 1.5 * IQR) | (length > Q3 + 1.5 * IQR)]
print("\nOutliers:\n", outliers)
