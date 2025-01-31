import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def simple_linear_regression(X, Y):
    X_mean, Y_mean = np.mean(X), np.mean(Y)
    num = np.sum((X - X_mean) * (Y - Y_mean))
    den = np.sum((X - X_mean) ** 2)
    m = num / den
    b = Y_mean - (m * X_mean)
    Y_pred = m * X + b
    return m, b, Y_pred

def multiple_linear_regression(X, Y):
    X = np.column_stack((np.ones(X.shape[0]), X))  
    theta = np.linalg.inv(X.T @ X) @ X.T @ Y 
    Y_pred = X @ theta
    return theta, Y_pred

file_path = "/content/sample_data/residency info.xlsx" 
df = pd.read_excel(file_path, sheet_name='in')


print("\nFirst 5 rows of dataset:\n", df.head())


print("\n--- SIMPLE LINEAR REGRESSION ---")


if 'Age' in df.columns and 'NetWorth' in df.columns:
    df_simple = df[['Age', 'NetWorth']].dropna()
    X_simple = df_simple['Age'].values
    Y_simple = df_simple['NetWorth'].values

    
    m, b, Y_pred_simple = simple_linear_regression(X_simple, Y_simple)

    
    mse_simple = mean_squared_error(Y_simple, Y_pred_simple)
    r2_simple = r2_score(Y_simple, Y_pred_simple)

  
    print(f"Equation: NetWorth = {m:.2f} * Age + {b:.2f}")
    print(f"Mean Squared Error: {mse_simple:.2f}")
    print(f"R-squared: {r2_simple:.4f}")

    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_simple, Y_simple, color='blue', label='Actual Data')
    plt.plot(X_simple, Y_pred_simple, color='red', label='Regression Line')
    plt.xlabel('Age')
    plt.ylabel('NetWorth')
    plt.title('Simple Linear Regression')
    plt.legend()
    plt.show()
else:
    print("Required columns for Simple Regression not found!")

print("\n--- MULTIPLE LINEAR REGRESSION ---")


if {'Age', 'NetWorth', 'Rank'}.issubset(df.columns):
    df_multiple = df[['Age', 'NetWorth', 'Rank']].dropna()
    X_multiple = df_multiple[['Age', 'Rank']].values
    Y_multiple = df_multiple['NetWorth'].values

  
    theta, Y_pred_multiple = multiple_linear_regression(X_multiple, Y_multiple)

  
    mse_multiple = mean_squared_error(Y_multiple, Y_pred_multiple)
    r2_multiple = r2_score(Y_multiple, Y_pred_multiple)

    
    print(f"Equation: NetWorth = {theta[0]:.2f} + ({theta[1]:.2f} * Age) + ({theta[2]:.2f} * Rank)")
    print(f"Mean Squared Error: {mse_multiple:.2f}")
    print(f"R-squared: {r2_multiple:.4f}")

  
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_multiple['Age'], df_multiple['Rank'], df_multiple['NetWorth'], color='blue', label='Actual Data')
    ax.set_xlabel('Age')
    ax.set_ylabel('Rank')
    ax.set_zlabel('NetWorth')
    ax.set_title('Multiple Linear Regression')2
    plt.show()
else:
    print("Required columns for Multiple Regression not found!")
