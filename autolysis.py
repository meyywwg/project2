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
    """Calculate skewness for numerical columns."""
    return numeric_columns.skew().to_string()

def perform_pca(numeric_columns):
    """Perform Principal Component Analysis (PCA) on numerical data."""
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(numeric_columns.dropna())
    explained_variance = pca.explained_variance_ratio_.tolist()
    return pca_result, explained_variance

def perform_clustering(numeric_columns):
    """Perform K-Means clustering on numerical data."""
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(numeric_columns.dropna())
    return clusters

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

def generate_prompt(base_folder, column_types, summary_stats, missing_values, outliers, skewness_values, pca_variance, clusters, correlation_matrix=None, additional_focus=""):
    """Generate a dynamic prompt for LLM with optional focus areas."""
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

{"### 8. Correlation Matrix:" if correlation_matrix is not None else ""}
{correlation_matrix.to_string() if correlation_matrix is not None else ""}

{f"### Additional Focus:\n{additional_focus}" if additional_focus else ""}

Please provide insights, trends, or recommendations based on these findings. Explain any unexpected correlations, patterns in missing data, and the rationale behind PCA and clustering results. Suggest methods to handle missing data and improve model performance if relevant.
"""
    return prompt

def choose_ai_model(data_analysis_type="text"):
    """Dynamically select which AI model to use based on data analysis type."""
    # Example of selecting a text analysis model or a vision model.
    if data_analysis_type == "text":
        return "gpt-4o-mini"
    elif data_analysis_type == "vision":
        return "openai-vision"
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
    """Save the analysis report and references to visualizations."""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Analysis Report\n\n")
        if story:
            f.write("## Story\n\n")
            f.write(story)
        else:
            f.write("## Story\n\nNo story generated.\n")
        f.write("\n\n## Visualizations\n")
        f.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")
        f.write("![Missing Values](missing_values.png)\n\n")
        f.write("![Distribution](distribution.png)\n")
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

    # Example of dynamic behavior: asking the user for additional focus area
    additional_focus = input("Enter any additional focus area (e.g., 'focus on PCA') or press Enter to skip: ")

    prompt = generate_prompt(base_folder, column_types, summary_stats, missing_values, outliers, skewness_values, pca_variance, clusters, correlation_matrix, additional_focus)

    # Choose AI model based on the task (text-based analysis here)
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
