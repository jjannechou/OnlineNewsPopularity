import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV



def load_data(filename):
    # Load the dataset
    data = pd.read_csv(filename)
    data = data.iloc[:, 2:]

    # Display basic information
    print("Basic Information about the Dataset:")
    print(data.info())

    # Check number of rows and columns
    print("No of Rows: {}".format(data.shape[0]))
    print("No of Columns: {}".format(data.shape[1]))

    # Display first few instances
    print("\n<Data View: First Few Instances>\n")
    print(data.head())
    
    return data



def plot_top_correlations(data, target_column, top_n=10):
    # Clean column names to ensure consistency
    data.columns = data.columns.str.strip().str.lower()
    target_column = target_column.strip().lower()

    # Select numeric columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    # Compute the correlation matrix
    correlation_matrix = data[numeric_columns].corr()

    # Identify top correlated features with the target column
    if target_column not in correlation_matrix.columns:
        raise ValueError(f"Column '{target_column}' not found in the dataset. Available columns: {correlation_matrix.columns}")
    
    top_corr_features = correlation_matrix[target_column].sort_values(ascending=False).head(top_n + 1).index
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Top {top_n} Correlations with '{target_column}'")
    plt.show()
    
    return list(top_corr_features)



def split_data(X_data, y_data):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=5)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=5)

    # Reset index
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Print splitting information
    print("\n************** Splitting Data **************\n")
    print(f"Train Data: ({len(X_train)}, {X_train.shape[1]})")
    print(f"Validation Data: ({len(X_val)}, {X_val.shape[1]})")
    print(f"Test Data: ({len(X_test)}, {X_test.shape[1]})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
