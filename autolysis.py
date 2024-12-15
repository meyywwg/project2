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
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Class for Data Loading
class DataLoader:
    @staticmethod
    def detect_encoding(filename):
        """Detect the file encoding for the given file."""
        with open(filename, 'rb') as f:
            raw_data = f.read()
        return chardet.detect(raw_data)['encoding']

    @staticmethod
    def load_dataset(filename):
        """Load dataset with detected encoding."""
        encoding = DataLoader.detect_encoding(filename)
        return pd.read_csv(filename, encoding=encoding)

    @staticmethod
    def create_directory_structure(base_folder):
        """Create the output directory dynamically."""
        output_dir = Path(base_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

# Class for Data Analysis
class DataAnalysis:
    def __init__(self, df):
        self.df = df
        self.numeric_columns = df.select_dtypes(include=["number"])

    def generate_summary(self):
        """Generate a summary of the dataset."""
        summary_stats = self.df.describe(include="all").to_string()
        missing_values = self.df.isnull().sum().to_string()
        column_types = self.df.dtypes.to_string()
        return summary_stats, missing_values, column_types

    def detect_outliers(self):
        """Detect outliers using the IQR method."""
        Q1 = self.numeric_columns.quantile(0.25)
        Q3 = self.numeric_columns.quantile(0.75)
        IQR = Q3 - Q1
        outliers = self.numeric_columns[(self.numeric_columns < Q1 - 1.5 * IQR) | (self.numeric_columns > Q3 + 1.5 * IQR)].dropna()
        return outliers

    def calculate_skewness(self):
        """Calculate skewness for numerical columns and test normality."""
        skewness = self.numeric_columns.skew().to_dict()
        normality_test = {
            col: shapiro(self.numeric_columns[col].dropna())[1]  # p-value of Shapiro-Wilk test
            for col in self.numeric_columns.columns
        }
        return skewness, normality_test

    def perform_pca(self):
        """Perform Principal Component Analysis (PCA) on numerical data."""
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.numeric_columns.dropna())
        explained_variance = pca.explained_variance_ratio_.tolist()
        loadings = pd.DataFrame(pca.components_, columns=self.numeric_columns.columns, index=['PC1', 'PC2'])
        return pca_result, explained_variance, loadings

    def perform_clustering(self):
        """Perform K-Means clustering and calculate silhouette score."""
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(self.numeric_columns.dropna())
        silhouette_avg = silhouette_score(self.numeric_columns.dropna(), clusters)
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=self.numeric_columns.columns)
        return clusters, silhouette_avg, cluster_centers

    def explore_correlations(self, correlation_matrix):
        """Identify and analyze unexpected correlations."""
        unexpected = correlation_matrix[(correlation_matrix > 0.8) & (correlation_matrix < 1.0)]
        if unexpected.empty:
            return "No unexpected high correlations found."
        analysis = """High correlations detected between the following features:\n"""
        for row in unexpected.iterrows():
            feature = row[0]
            correlations = row[1][row[1] > 0.8]
            for correlated_feature, value in correlations.items():
                analysis += f"- {feature} and {correlated_feature}: {value:.2f}\n"
        return analysis

# Class for Visualizations
class Visualizations:
    @staticmethod
    def create_visualizations(df, numeric_columns, correlation_matrix, output_dir):
        """Generate and save visualizations with detailed headings and captions."""
        visuals = {}

        # Correlation Heatmap
        if correlation_matrix is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.xlabel("Features")
            plt.ylabel("Features")
            heatmap_path = output_dir / "correlation_heatmap.png"
            plt.savefig(heatmap_path)
            visuals["Correlation Heatmap"] = heatmap_path

        # Missing Values
        missing_values_series = df.isnull().sum()
        if missing_values_series.sum() > 0:
            plt.figure(figsize=(8, 6))
            missing_values_series[missing_values_series > 0].plot(kind="bar", color="skyblue")
            plt.title("Missing Values per Column")
            plt.xlabel("Columns")
            plt.ylabel("Count of Missing Values")
            missing_values_path = output_dir / "missing_values.png"
            plt.savefig(missing_values_path)
            visuals["Missing Values"] = missing_values_path

        # Distribution of First Numerical Column
        if not numeric_columns.empty:
            plt.figure(figsize=(8, 6))
            sns.histplot(numeric_columns.iloc[:, 0].dropna(), kde=True, color="purple")
            plt.title(f"Distribution of {numeric_columns.columns[0]}")
            plt.xlabel(numeric_columns.columns[0])
            plt.ylabel("Frequency")
            dist_path = output_dir / "distribution.png"
            plt.savefig(dist_path)
            visuals["Distribution"] = dist_path

        # Boxplot for Numeric Columns
        if not numeric_columns.empty:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=numeric_columns)
            plt.title("Boxplot of Numerical Columns")
            plt.xlabel("Columns")
            plt.ylabel("Values")
            boxplot_path = output_dir / "boxplot.png"
            plt.savefig(boxplot_path)
            visuals["Boxplot"] = boxplot_path

        return visuals

# Class for AI Interaction
class AIInteraction:
    @staticmethod
    def get_aiproxy_token():
        """Retrieve AIPROXY_TOKEN from environment variables."""
        token = os.getenv("AIPROXY_TOKEN")
        if not token:
            raise ValueError("AIPROXY_TOKEN is not set in the .env file!")
        return token

    @staticmethod
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
            "max_tokens": 600,
            "temperature": 0.7
        }

        response = requests.post(proxy_url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(f"Error: {response.status_code}, {response.text}")

    @staticmethod
    def choose_ai_model(data_analysis_type="text", complexity="basic"):
        """Dynamically select which AI model to use based on data analysis complexity."""
        if data_analysis_type == "text" and complexity == "basic":
            return "gpt-4o-mini"
        elif data_analysis_type == "text" and complexity == "advanced":
            return "gpt-4"
        elif data_analysis_type == "code" and complexity == "basic":
            return "code-davinci-002"
        elif data_analysis_type == "code" and complexity == "advanced":
            return "code-davinci-003"
        else:
            return "gpt-4o-mini"

# Main Execution
def main():
    aiproxy_token = AIInteraction.get_aiproxy_token()

    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    base_folder = Path(filename).stem

    output_dir = DataLoader.create_directory_structure(base_folder)
    readme_path = output_dir / "README.md"

    try:
        df = DataLoader.load_dataset(filename)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)

    data_analysis = DataAnalysis(df)
    summary_stats, missing_values, column_types = data_analysis.generate_summary()
    correlation_matrix = df.select_dtypes(include=["number"]).corr() if len(df.select_dtypes(include=["number"]).columns) > 1 else None
    outliers = data_analysis.detect_outliers()
    skewness_values, normality_test = data_analysis.calculate_skewness()
    pca_result, pca_variance, pca_loadings = data_analysis.perform_pca()
    clusters, silhouette_avg, cluster_centers = data_analysis.perform_clustering()
    correlation_analysis = data_analysis.explore_correlations(correlation_matrix)

    prompt = f"""
    Dataset Analysis for {base_folder}:
    Column Types:
    {column_types}

    Summary Statistics (Key Metrics):
    {summary_stats}

    Notable Observations:
    - Missing Values: {missing_values}
    - Outliers Detected: {len(outliers)}
    - Skewness: {skewness_values}

    PCA Explained Variance: {pca_variance}
    Clustering Silhouette Score: {silhouette_avg}

    Correlation Insights:
    {correlation_matrix.to_string() if correlation_matrix is not None else "Not enough data for correlations."}
    """
    
    ai_model = AIInteraction.choose_ai_model(data_analysis_type="text", complexity="basic")

    try:
        story = AIInteraction.send_to_ai_proxy(prompt, aiproxy_token, model=ai_model)
    except RuntimeError as e:
        print(str(e))
        story = "Error generating the story. Please check the AI Proxy."

    visuals = Visualizations.create_visualizations(df, df.select_dtypes(include=["number"]), correlation_matrix, output_dir)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Analysis Report\n\n")
        f.write("## Story\n\n")
        f.write(story if story else "No story generated.\n")
        f.write("\n\n## Visualizations\n")
        for title, path in visuals.items():
            f.write(f"### {title}\n![{title}]({path})\n")

if __name__ == "__main__":
    main()



