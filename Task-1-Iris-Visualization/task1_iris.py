import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris Dataset
iris = sns.load_dataset("iris")

# Dataset Structure
print("Shape:", iris.shape)

print("\nColumns:")
print(iris.columns)

print("\nFirst 5 Rows:")
print(iris.head())

# Scatter Plot
sns.scatterplot(
    data=iris,
    x="sepal_length",
    y="petal_length",
    hue="species"
)

plt.title("Sepal Length vs Petal Length")
plt.show()

# Histogram
plt.figure(figsize=(8, 5))

sns.histplot(
    iris["sepal_width"],
    bins=15,
    kde=True
)

plt.title("Distribution of Sepal Width")
plt.show()

# Box Plot
plt.figure(figsize=(8, 5))

sns.boxplot(data=iris)

plt.title("Feature Distribution and Outliers")
plt.show()