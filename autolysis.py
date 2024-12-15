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
from sklearn.metrics import silhouette_score
from scipy.stats import shapiro

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
    """Load dataset with detected encoding."""
    encoding = detect_encoding(filename)
    return pd.read_csv(filename, encoding=encoding)

def create_directory_structure(base_folder):
    """Create the output directory dynamically."""
    output_dir = Path(base_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def generate_summary(df):
    """Generate a summary of the dataset."""
    summary_stats = df.describe(include="all").to_string()
    missing_values = df.isnull().sum().to_string()
    column_types = df.dtypes.to_string()
    return summary_stats, missing_values, column_types

def detect_outliers(numeric_columns):
    """Detect outliers using the IQR method."""
    Q1 = numeric_columns.quantile(0.25)
    Q3 = numeric_columns.quantile(0.75)
    IQR = Q3 - Q1
    outliers = numeric_columns[(numeric_columns < Q1 - 1.5 * IQR) | (numeric_columns > Q3 + 1.5 * IQR)].dropna()
    return outliers

def calculate_skewness(numeric_columns):
    """Calculate skewness for numerical columns and test normality."""
    skewness = numeric_columns.skew().to_dict()
    normality_test = {
        col: shapiro(numeric_columns[col].dropna())[1]  # p-value of Shapiro-Wilk test
        for col in numeric_columns.columns
    }
    return skewness, normality_test

def perform_pca(numeric_columns):
    """Perform Principal Component Analysis (PCA) on numerical data."""
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(numeric_columns.dropna())
    explained_variance = pca.explained_variance_ratio_.tolist()
    loadings = pd.DataFrame(pca.components_, columns=numeric_columns.columns, index=['PC1', 'PC2'])
    return pca_result, explained_variance, loadings

def perform_clustering(numeric_columns):
    """Perform K-Means clustering and calculate silhouette score."""
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(numeric_columns.dropna())
    silhouette_avg = silhouette_score(numeric_columns.dropna(), clusters)
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_columns.columns)
    return clusters, silhouette_avg, cluster_centers

def create_visualizations(df, numeric_columns, correlation_matrix, output_dir):
    """Generate and save visualizations."""
    # Visualization 1: Correlation Heatmap
    if correlation_matrix is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.xlabel("Features")
        plt.ylabel("Features")
        heatmap_path = output_dir / "correlation_heatmap.png"
        plt.savefig(heatmap_path)

    # Visualization 2: Missing Values
    missing_values_series = df.isnull().sum()
    if missing_values_series.sum() > 0:
        plt.figure(figsize=(8, 6))
        missing_values_series[missing_values_series > 0].plot(kind="bar", color="skyblue")
        plt.title("Missing Values per Column")
        plt.xlabel("Columns")
        plt.ylabel("Count of Missing Values")
        missing_values_path = output_dir / "missing_values.png"
        plt.savefig(missing_values_path)

    # Visualization 3: Distribution of First Numerical Column
    if not numeric_columns.empty:
        plt.figure(figsize=(8, 6))
        sns.histplot(numeric_columns.iloc[:, 0].dropna(), kde=True, color="purple")
        plt.title(f"Distribution of {numeric_columns.columns[0]}")
        plt.xlabel(numeric_columns.columns[0])
        plt.ylabel("Frequency")
        dist_path = output_dir / "distribution.png"
        plt.savefig(dist_path)

    # Visualization 4: Boxplot for Numeric Columns
    if not numeric_columns.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=numeric_columns)
        plt.title("Boxplot of Numerical Columns")
        plt.xlabel("Columns")
        plt.ylabel("Values")
        boxplot_path = output_dir / "boxplot.png"
        plt.savefig(boxplot_path)

def generate_prompt(base_folder, column_types, summary_stats, missing_values, outliers, skewness_values, normality_test, pca_variance, pca_loadings, clusters, silhouette_avg, cluster_centers, correlation_matrix=None):
    """Generate a dynamic prompt for LLM."""
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

### 6. Normality Test (p-values):
{normality_test}

### 7. PCA Explained Variance:
{pca_variance}

### 8. PCA Loadings:
{pca_loadings}

### 9. Clustering Results:
Silhouette Score: {silhouette_avg}
Cluster Centers:
{cluster_centers}

{f"### 10. Correlation Matrix:\n{correlation_matrix.to_string()}" if correlation_matrix is not None else ""}

Please provide insights, trends, or recommendations based on these findings. Explain any unexpected correlations, patterns in missing data, and the rationale behind PCA and clustering results. Suggest methods to handle missing data and improve model performance if relevant.
"""
    return prompt

def choose_ai_model(data_analysis_type="text", complexity="basic"):
    """Dynamically select which AI model to use based on data analysis complexity."""
    if data_analysis_type == "text":
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
    skewness_values, normality_test = calculate_skewness(numeric_columns)
    pca_result, pca_variance, pca_loadings = perform_pca(numeric_columns)
    clusters, silhouette_avg, cluster_centers = perform_clustering(numeric_columns)

    # Generate prompt
    prompt = generate_prompt(base_folder, column_types, summary_stats, missing_values, outliers, skewness_values, normality_test, pca_variance, pca_loadings, clusters, silhouette_avg, cluster_centers, correlation_matrix)

    # Choose AI model
    ai_model = choose_ai_model(data_analysis_type="text")

    try:
        story = send_to_ai_proxy(prompt, aiproxy_token, model=ai_model)
    except RuntimeError as e:
        print(str(e))
        story = "Error generating the story. Please check the AI Proxy."

    save_analysis(readme_path, story, output_dir)
    create_visualizations(df, numeric_columns, correlation_matrix, output_dir)

if __name__ == "__main__":
    main()
