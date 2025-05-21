import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_excel('Data for Task 1.xlsx')
print(df)

# 1.Column-wise analysis
def column_wise_analysis(df):
    analysis_result = []
    for column in df.columns:
        column_data = df[column]
        unique_vals = column_data.nunique(dropna=True)
        value_counts = column_data.value_counts(dropna=False).head(10).to_dict()
        significance = f"Significance of '{column}' for stakeholders."
        column_info = {
            'Column Name': column,
            'Data Type': str(column_data.dtype),
            'Unique Values': unique_vals,
            'Top Value Counts (max 10)': value_counts,
            'Significance': significance
        }
        analysis_result.append(column_info)
    return pd.DataFrame(analysis_result)

# 2a. Handle Missing or Invalid Values
def handle_missing_values(df):
    # Missing data percentage per column
    missing_percent = df.isnull().mean() * 100
    print("Missing data percentage by column:")
    print(missing_percent)

    # Drop columns with >50% missing
    cols_to_drop = missing_percent[missing_percent > 50].index.tolist()
    if cols_to_drop:
        print(f"Dropping columns with >50% missing values: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Impute or drop for remaining missing data
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Fill categorical with mode(the most frequently occurring value) of the column
    for col in categorical_cols:
        mode_val = df[col].mode()
        if not mode_val.empty:
            # Fill any missing values in column with mode value
            df[col].fillna(mode_val[0], inplace=True)

    # Fill numerical with median(the middle value when the data is sorted) of the column
    for col in numerical_cols:
        median_val = df[col].median()
        # Fill any missing values in column with median value
        df[col].fillna(median_val, inplace=True)

    # Drop any remaining rows with at least one missing value (if any)
    df.dropna(inplace=True)
    return df

# 2b. Address inconsistencies in categorical data
def clean_categorical_columns(df):
    categorical_cols = df.select_dtypes(include=['object'])
    for col in categorical_cols.columns:
        # Convert to string, strip whitespace and standardize capitalization
        df[col] = df[col].astype(str).str.strip().str.title()
    return df

# 2c. Fix numerical columns format and outliers
def handle_numerical_columns(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        # Ensure numeric type
        df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle outliers via IQR capping
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    return df

# 3a. Identifying Critical Columns
def identify_critical_columns(df):
    # Example criteria for critical columns
    critical_columns = ['VIN', 'REPAIR_DATE', 'COMPLAINT_CD', 'GLOBAL_LABOR_CODE', 'TOTALCOST']
    return critical_columns

# 3c. Visualizations
def generate_visualizations(df):
    # Visualization 1: Distribution of Total Costs
    plt.figure(figsize=(10, 6))
    sns.histplot(df['TOTALCOST'], bins=30, kde=True)
    plt.title('Distribution of Total Repair Costs')
    plt.xlabel('Total Cost')
    plt.ylabel('Frequency')
    plt.savefig('total_cost_distribution.png')
    plt.show()

    # Visualization 2: Count of Unique Complaints
    plt.figure(figsize=(10, 6))
    complaint_counts = df['COMPLAINT_CD'].value_counts().head(10)
    sns.barplot(x=complaint_counts.index, y=complaint_counts.values)
    plt.title('Top 10 Unique Complaints')
    plt.xlabel('Complaint Code')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('top_complaints.png')
    plt.show()

    # Visualization 3: Repair Counts by Repair Date
    df['REPAIR_DATE'] = pd.to_datetime(df['REPAIR_DATE'])
    repair_counts = df['REPAIR_DATE'].dt.to_period('M').value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=repair_counts.index.astype(str), y=repair_counts.values)
    plt.title('Number of Repairs Over Time')
    plt.xlabel('Repair Date (Month)')
    plt.ylabel('Number of Repairs')
    plt.xticks(rotation=45)
    plt.savefig('repairs_over_time.png')
    plt.show()

# 4. Generating Tags/Features from Free Text
def generate_tags(df):
    free_text_columns = ['CORRECTION_VERBATIM', 'CUSTOMER_VERBATIM']
    tags = []

    for col in free_text_columns:
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df[col].dropna())
        tags.extend(vectorizer.get_feature_names_out())

    return set(tags)

# 5. Overall Synthesis/Key Takeaways
def overall_synthesis(df):
    # Highlight discrepancies
    discrepancies = df.isnull().sum()
    print("Discrepancies in the dataset (missing values):")
    print(discrepancies)

    # Summary of generated tags
    tags = generate_tags(df)
    print("Generated Tags:")
    print(tags)

    # Recommendations based on analysis
    recommendations = [
        "Focus on improving the quality of parts that frequently require repairs.",
        "Enhance customer service training based on common complaints.",
        "Monitor repair costs closely to identify potential areas for cost savings."
    ]
    print("Recommendations for stakeholders:")
    for rec in recommendations:
        print(f"- {rec}")

def main():
    print("Starting column-wise analysis...")
    analysis_df = column_wise_analysis(df)
    print("Column-wise analysis completed.")

    # Save analysis to CSV for review
    analysis_df.to_csv('column_wise_analysis.csv', index=False)
    print("Column-wise analysis saved to 'column_wise_analysis.csv'.")

    print("Starting data cleaning...")
    df_cleaned = handle_missing_values(df)
    df_cleaned = clean_categorical_columns(df_cleaned)
    df_cleaned = handle_numerical_columns(df_cleaned)
    print("Data cleaning completed.")

    # Save cleaned data
    df_cleaned.to_excel('vehicle_repairs_cleaned.xlsx', index=False)
    print("Cleaned dataset saved.")

    # Identify critical columns
    critical_columns = identify_critical_columns(df_cleaned)
    print("Identified Critical Columns:")
    print(critical_columns)

    # Generate visualizations
    generate_visualizations(df_cleaned)

    # Overall synthesis and key takeaways
    overall_synthesis(df_cleaned)


if __name__ == "__main__":
    main()

