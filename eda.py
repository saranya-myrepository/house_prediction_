import os
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
from scipy.stats import chi2
from numpy.linalg import LinAlgError
import statsmodels.api as sm
import pickle
import seaborn as sns
import sys
sys.stdout = open("output_log.txt", "w")


# Print current working directory to console before redirection
print("Current working directory:", os.getcwd())

# Redirect all print output to file:
with open("output_log.txt", "w") as f:
    sys.stdout = f  # redirect print

    print("Redirected print output to output_log.txt")

    # ------------------ Load & Explore Data ------------------
    df = pd.read_csv("data/Melbourne_housing_FULL.csv")

    # Show first 5 rows
    print(df.head())

    # Show numeric summary
    print(df.describe())

    # Show last 3 rows
    print(df.tail(3))

    # Show shape
    print(df.shape)

    # Count of missing values
    missing_count = df.isnull().sum()
    print("Missing value counts:")
    print(missing_count)

    # Count of zero values in numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    zero_counts = (df_numeric == 0).sum()
    print("\nThe zero_counts of the data is ")
    print(zero_counts)

    # ------------------ Visualize Missingness ------------------
    print("\nShowing missing data matrix...")
    msno.matrix(df)
    plt.show()

    print("\nShowing missing data heatmap...")
    msno.heatmap(df)
    plt.show()

    # ------------------ Little's MCAR Test ------------------
    def little_mcar(df):
        df = df.copy()
        df = df.select_dtypes(include=[np.number])  # Only numeric columns
        df = df.dropna(axis=1, how='all')  # Drop fully empty columns
        df = df.dropna(axis=0, how='all')  # Drop fully empty rows

        patterns = df.isnull().astype(int)
        pattern_groups = patterns.groupby(list(df.columns)).size().reset_index(name='count')

        chi_square = 0
        df_total = 0

        complete_cases = df.dropna()
        if len(complete_cases) == 0:
            raise ValueError("No complete cases available for MCAR test.")

        mu = complete_cases.mean()
        cov = complete_cases.cov()

        for _, row in pattern_groups.iterrows():
            pattern = row[:-1]
            count = row['count']
            match = (df.isnull() == pattern).all(axis=1)
            subgroup = df[match]

            observed_cols = df.columns[[not is_missing for is_missing in pattern]]
            sub_data = subgroup[observed_cols].dropna()

            if len(sub_data) == 0:
                continue

            mu_g = mu[observed_cols]
            cov_g = cov.loc[observed_cols, observed_cols]

            try:
                inv_cov_g = np.linalg.inv(cov_g)
            except LinAlgError:
                continue  # Skip ill-conditioned groups

            centered = sub_data - mu_g
            d2 = np.einsum('ij,jk,ik->i', centered.values, inv_cov_g, centered.values)
            chi_square += d2.sum()
            df_total += len(observed_cols) * len(sub_data)

        p_value = 1 - chi2.cdf(chi_square, df_total)

        return {
            "chi_square": round(chi_square, 3),
            "df": df_total,
            "p_value": round(p_value, 5)
        }

    # Run MCAR test
    result = little_mcar(df)

    print("\nLittle's MCAR Test Results:")
    print(f"Chi-Square: {result['chi_square']}")
    print(f"Degrees of Freedom: {result['df']}")
    print(f"P-Value: {result['p_value']}")

    if result["p_value"] > 0.05:
        print("Data is likely MCAR (Missing Completely At Random)")
    else:
        print("Data is NOT MCAR (Missing Completely At Random)")
    print(df[['Landsize', 'Price']].head(15))
    print(df.select_dtypes(include=['number']).columns)
    print(df.select_dtypes(include=['object']).columns)

    # for mar test of missing values
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for target_col in numeric_cols:
        if df[target_col].isnull().sum() == 0:
            continue
        print(f"\n--- Testing Missingness for: {target_col} ---")
        for target_col in numeric_cols:
            if df[target_col].isnull().sum() == 0:
                continue  # Skip if no missing data in this column

            print(f"\n--- Testing Missingness for: {target_col} ---")

            # Create missing indicator: 1 = missing, 0 = not missing
            df[f'missing_{target_col}'] = df[target_col].isnull().astype(int)

            # Select predictors (other numerical columns without the target)
            predictors = [col for col in numeric_cols if col != target_col]

            # Drop rows where predictors are missing
            model_data = df[predictors + [f'missing_{target_col}']].dropna()

            if model_data[f'missing_{target_col}'].nunique() < 2:
                print("❌ Not enough variation in missingness to fit model.")
                continue

            X = model_data[predictors]
            y = model_data[f'missing_{target_col}']

            # Add intercept
            X = sm.add_constant(X)

            try:
                model = sm.Logit(y, X).fit(disp=False)
                print(model.summary())
                print("\n  Interpretation:")
                print("   - Significant (p < 0.05) predictors → Missingness may be MAR")
                print("   - Non-significant → Missingness may be MCAR or MNAR (not conclusive)\n")
            except Exception as e:
                print(f" Could not fit model for {target_col}: {e}")
    print(df.isna().sum())
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer

    print("\nMissing value counts before imputation:")
    print(df.isna().sum())

    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Initialize IterativeImputer (MICE)
    iter_imputer = IterativeImputer(random_state=42)

    # Fit and transform numeric columns with missing data
    df[numeric_cols] = iter_imputer.fit_transform(df[numeric_cols])

    print("\nMissing value counts after imputation:")
    print(df.isna().sum())
    df.to_csv("Melbourne_housing_imputed.csv", index=False)
    df = df.dropna(subset=['CouncilArea', 'Regionname'])
    print("Missing values after dropping rows:")
    print(df.isna().sum())
    numeric_cols = df.select_dtypes(include=['number']).columns

    outliers = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outlier_mask.sum()
        outliers[col] = outlier_count

    print("Number of outliers detected per numeric column:")
    for col, count in outliers.items():
        print(f"{col}: {count}")
    df.drop(columns=[
        'missing_Price',
        'missing_Distance',
        'missing_Postcode',
        'missing_Bedroom2',
        'missing_Bathroom',
        'missing_Car',
        'missing_Landsize',
        'missing_BuildingArea',
        'missing_YearBuilt',
        'missing_Lattitude',
        'missing_Longtitude',
        'missing_Propertycount'
    ], inplace=True)

    # Confirm they are dropped
    print("Columns after dropping specified missing_ columns:")
    print(df.columns)
    # List of missing indicator columns
    cols_to_drop = [
        'missing_Price', 'missing_Distance', 'missing_Postcode',
        'missing_Bedroom2', 'missing_Bathroom', 'missing_Car',
        'missing_Landsize', 'missing_BuildingArea', 'missing_YearBuilt',
        'missing_Lattitude', 'missing_Longtitude', 'missing_Propertycount'
    ]

    # Drop only if column exists
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    # For finding outliers
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Dictionary to store outlier counts
    outliers = {}

    # Loop through each numeric column to calculate IQR and outlier bounds
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers[col] = outlier_count

    # Print the count of outliers
    print("Number of outliers detected per numeric column:")
    for col, count in outliers.items():
        print(f"{col}: {count}")
    print(df.corr(numeric_only=True)['Price'].sort_values(ascending=False))

    # visualize outliers per feature
    # Get only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Plot boxplot for each numeric column
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.show()

    # Remove outliers (optional step)
    Q1 = df['Landsize'].quantile(0.25)
    Q3 = df['Landsize'].quantile(0.75)
    IQR = Q3 - Q1
    filtered_df = df[(df['Landsize'] > Q1 - 1.5 * IQR) & (df['Landsize'] < Q3 + 1.5 * IQR)]

    # Now check correlation again
    print(filtered_df[['Price', 'Landsize']].corr())

    df['Landsize_log'] = np.log1p(df['Landsize'])  # log1p handles 0 valu
    categorical_cols = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Date', 'CouncilArea', 'Regionname']

    for col in categorical_cols:
        unique_vals = df[col].nunique()
        print(f"{col}: {unique_vals} unique values")
    columns_to_drop = ['Address', 'SellerG', 'Postcode', 'Suburb', 'Lattitude', 'Longtitude']
    df = df.drop(columns=columns_to_drop)
    from sklearn.preprocessing import LabelEncoder
    cols_to_encode = ['Type', 'Method', 'CouncilArea', 'Regionname']
    label_encoders = {}
    for col in cols_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    print("Label-encoded dataframe:")
    print(df.head())
    print(df[['Rooms', 'Bedroom2']].corr())
    df = df.drop(columns=['Bedroom2'])


