# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "python-dotenv",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "scikit-learn",
#   "chardet",
#   "requests",
# ]
# ///
from dotenv import load_dotenv
import os
import pandas as pd
import chardet
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import requests
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load environment variables from the .env file
load_dotenv()

def get_aiproxy_token():
    """Retrieve AIPROXY_TOKEN from environment variables."""
    token = os.getenv("AIPROXY_TOKEN")
    if not token:
        raise ValueError("AIPROXY_TOKEN is not set in the .env file!")
    return token

def detect_encoding(filename):
    """Detect the file encoding for the given file."""
    with open(filename, 'rb') as f:
        raw_data = f.read()
    return chardet.detect(raw_data)['encoding']

def load_dataset(filename):
    """Load dataset with detected encoding, with error handling."""
    try:
        encoding = detect_encoding(filename)
        return pd.read_csv(filename, encoding=encoding)
    except Exception as e:
        raise RuntimeError(f"Error loading {filename}: {e}")

def create_directory_structure(base_folder):
    """Create the output directory dynamically."""
    output_dir = Path(base_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def generate_summary(df):
    """Generate a summary of the dataset, handling missing data."""
    summary_stats = df.describe(include="all").to_string()
    missing_values = df.isnull().sum().to_string()
    column_types = df.dtypes.to_string()
    return summary_stats, missing_values, column_types

def detect_outliers(numeric_columns):
    """Detect outliers using the IQR method, with error handling."""
    try:
        Q1 = numeric_columns.quantile(0.25)
        Q3 = numeric_columns.quantile(0.75)
        IQR = Q3 - Q1
        outliers = numeric_columns[(numeric_columns < Q1 - 1.5 * IQR) | (numeric_columns > Q3 + 1.5 * IQR)].dropna()
        return outliers
    except Exception as e:
        raise RuntimeError(f"Error detecting outliers: {e}")

def calculate_skewness(numeric_columns):
    """Calculate skewness for numerical columns with error handling."""
    try:
        return numeric_columns.skew().to_string()
    except Exception as e:
        raise RuntimeError(f"Error calculating skewness: {e}")

def perform_pca(numeric_columns):
    """Perform PCA on numerical data with error handling."""
    try:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_columns.dropna())
        explained_variance = pca.explained_variance_ratio_.tolist()
        return pca_result, explained_variance
    except Exception as e:
        raise RuntimeError(f"Error performing PCA: {e}")

def perform_clustering(numeric_columns):
    """Perform K-Means clustering with error handling."""
    try:
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(numeric_columns.dropna())
        return clusters
    except Exception as e:
        raise RuntimeError(f"Error performing clustering: {e}")

def create_visualizations(df, numeric_columns, correlation_matrix, output_dir):
    """Generate and save visualizations with enhanced formatting and clarity."""
    try:
        # Correlation Heatmap
        if correlation_matrix is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar_kws={'label': 'Correlation Coefficient'})
            plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
            plt.xlabel("Features", fontsize=12)
            plt.ylabel("Features", fontsize=12)
            plt.legend(['Correlation Coefficients'], loc="upper right")
            heatmap_path = output_dir / "correlation_heatmap.png"
            plt.savefig(heatmap_path)

        # Missing Values
        missing_values_series = df.isnull().sum()
        if missing_values_series.sum() > 0:
            plt.figure(figsize=(8, 6))
            missing_values_series[missing_values_series > 0].plot(kind="bar", color="skyblue")
            plt.title("Missing Values per Column", fontsize=14, fontweight='bold')
            plt.xlabel("Columns", fontsize=12)
            plt.ylabel("Count of Missing Values", fontsize=12)
            plt.legend(['Missing Value Counts'], loc="upper right")
            missing_values_path = output_dir / "missing_values.png"
            plt.savefig(missing_values_path)

        # Distribution of First Numerical Column
        if not numeric_columns.empty:
            plt.figure(figsize=(8, 6))
            sns.histplot(numeric_columns.iloc[:, 0].dropna(), kde=True, color="purple")
            plt.title(f"Distribution of {numeric_columns.columns[0]}", fontsize=14, fontweight='bold')
            plt.xlabel(numeric_columns.columns[0], fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.legend(["Density", "Histogram"], loc="upper right")
            dist_path = output_dir / "distribution.png"
            plt.savefig(dist_path)

        # Boxplot for Numeric Columns
        if not numeric_columns.empty:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=numeric_columns, palette="Set3")
            plt.title("Boxplot of Numerical Columns", fontsize=14, fontweight='bold')
            plt.xlabel("Columns", fontsize=12)
            plt.ylabel("Values", fontsize=12)
            plt.legend(["Values per Column"], loc="upper right")
            boxplot_path = output_dir / "boxplot.png"
            plt.savefig(boxplot_path)

    except Exception as e:
        raise RuntimeError(f"Error generating visualizations: {e}")

def generate_prompt(base_folder, column_types, summary_stats, missing_values, outliers, skewness_values, pca_variance, clusters, correlation_matrix=None):
    """Generate a dynamic prompt for LLM with emphasis on insights and narrative."""
    prompt = f"""
    Analyze the following dataset summary for {base_folder}:

    ### 1. Column Types:
    {column_types}

    ### 2. Summary Statistics:
    {summary_stats}

    ### 3. Missing Values:
    {missing_values}

    ### 4. Outliers:
    {outliers}

    ### 5. Skewness:
    {skewness_values}

    ### 6. PCA Explained Variance:
    {pca_variance}

    ### 7. Clustering Results:
    {clusters}

    ### 8. Correlation Matrix:
    {correlation_matrix.to_string() if correlation_matrix is not None else "N/A"}

    ### 9. Visualizations:
    Please refer to the following visualizations:
    - **Correlation Heatmap**: Represents the correlation between numerical features.
    - **Missing Values**: A bar chart showing the count of missing values per column.
    - **Distribution**: Distribution of the first numerical column.
    - **Boxplot**: Displays the distribution and outliers in the numerical columns.

    ### Objective:
    Provide a coherent narrative summarizing the dataset's structure, trends, outliers, and correlations. Highlight key insights and their implications, focusing on actionable conclusions.
    """
    return prompt

def choose_ai_model(data_analysis_type="text", complexity="basic"):
    """Dynamically select which AI model to use based on data analysis complexity."""
    if complexity == "advanced":
        return "gpt-4-vision"
    elif data_analysis_type == "text":
        return "gpt-4o-mini"
    else:
        return "gpt-4o-mini"

def send_to_ai_proxy(prompt, aiproxy_token, model="gpt-4o-mini"):
    """Send the prompt to AI Proxy and return the generated response."""
    proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {aiproxy_token}"}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    response = requests.post(proxy_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"Error: {response.status_code}, {response.text}")

def save_analysis(readme_path, story, output_dir):
    """Save the analysis report with visualizations and a coherent narrative."""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Analysis Report\n\n")
        if story:
            f.write("## Story\n\n")
            f.write(story)
        else:
            f.write("## Story\n\nNo story generated.\n")
        f.write("\n\n## Visualizations\n")

        f.write("### Correlation Heatmap\n")
        f.write("This heatmap shows the correlation between different numerical features. Strong positive or negative correlations can reveal relationships worth exploring further.\n")
        f.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")

        f.write("### Missing Values\n")
        f.write("The bar chart highlights columns with missing values and their respective counts. This helps identify areas requiring data cleaning.\n")
        f.write("![Missing Values](missing_values.png)\n\n")

        f.write("### Distribution\n")
        f.write(f"This graph illustrates the distribution of the first numerical column ({numeric_columns.columns[0]}). It provides insights into the data's spread and central tendencies.\n")
        f.write("![Distribution](distribution.png)\n\n")

        f.write("### Boxplot\n")
        f.write("This boxplot visualizes the distribution of values and highlights potential outliers for each numerical column.\n")
        f.write("![Boxplot](boxplot.png)\n")

def main():
    aiproxy_token = get_aiproxy_token()

    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    base_folder = Path(filename).stem

    output_dir = create_directory_structure(base_folder)
    readme_path = output_dir / "README.md"

    try:
        df = load_dataset(filename)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)

    summary_stats, missing_values, column_types = generate_summary(df)
    numeric_columns = df.select_dtypes(include=["number"])
    correlation_matrix = numeric_columns.corr() if len(numeric_columns.columns) > 1 else None
    outliers = detect_outliers(numeric_columns)
    skewness_values = calculate_skewness(numeric_columns)
    pca_result, pca_variance = perform_pca(numeric_columns)
    clusters = perform_clustering(numeric_columns)

    prompt = generate_prompt(base_folder, column_types, summary_stats, missing_values, outliers, skewness_values, pca_variance, clusters, correlation_matrix)

    ai_model = choose_ai_model(data_analysis_type="text", complexity="advanced")

    try:
        story = send_to_ai_proxy(prompt, aiproxy_token, model=ai_model)
    except RuntimeError as e:
        print(str(e))
        story = "Error generating the story. Please check the AI Proxy."

    save_analysis(readme_path, story, output_dir)
    create_visualizations(df, numeric_columns, correlation_matrix, output_dir)

if __name__ == "__main__":
    main()

