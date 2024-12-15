# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "python-dotenv",
#   "pandas",
#   "matplotlib",
#   "seaborn",
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

# Load environment variables from the .env file
load_dotenv()

# Retrieve the AIPROXY_TOKEN
aiproxy_token = os.getenv("AIPROXY_TOKEN")
if not aiproxy_token:
    raise ValueError("AIPROXY_TOKEN is not set in the .env file!")

if len(sys.argv) != 2:
    print("Usage: python script.py <dataset.csv>")
    sys.exit(1)

filename = sys.argv[1]
base_folder = Path(filename).stem  # Use the stem (filename without extension) as the folder name


# Create the directory structure dynamically
output_dir = Path(base_folder)
output_dir.mkdir(parents=True, exist_ok=True)

# Detect encoding
with open(filename, 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    detected_encoding = result['encoding']

try:
    df = pd.read_csv(filename, encoding=detected_encoding)  # Use detected encoding
except Exception as e:
    print(f"Error loading {filename}: {e}")
    sys.exit(1)

# Inspect the dataset
summary_stats = df.describe(include="all").to_string()
missing_values = df.isnull().sum().to_string()
column_types = df.dtypes.to_string()

numeric_columns = df.select_dtypes(include=["number"])
correlation_matrix = numeric_columns.corr() if len(numeric_columns.columns) > 1 else None

# Detect outliers
Q1 = numeric_columns.quantile(0.25)
Q3 = numeric_columns.quantile(0.75)
IQR = Q3 - Q1
outliers = numeric_columns[(numeric_columns < Q1 - 1.5 * IQR) | (numeric_columns > Q3 + 1.5 * IQR)]

# Construct the AI prompt
prompt = f"""
Analyze the following dataset summary for {base_folder}:
1. Column Types:
{column_types}

2. Summary Statistics:
{summary_stats}

3. Missing Values:
{missing_values}

4. Outliers:
{outliers}

{"5. Correlation Matrix:" if correlation_matrix is not None else ""} 
{correlation_matrix.to_string() if correlation_matrix is not None else ""} 
Write a story summarizing the dataset, key findings, and actionable insights.
"""

# Send the prompt to AI Proxy
proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {aiproxy_token}"}

payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 500,
    "temperature": 0.7
}

response = requests.post(proxy_url, headers=headers, json=payload)
# Handle the response
if response.status_code == 200:
    story = response.json()["choices"][0]["message"]["content"]
else:
    print(f"Error: {response.status_code}, {response.text}")
    story = "Error generating the story. Please check the AI Proxy."


# Save story and visualizations
readme_path = output_dir / "README.md"
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



# Visualization 1: Correlation Heatmap
if correlation_matrix is not None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    heatmap_path = output_dir / "correlation_heatmap.png"
    plt.savefig(heatmap_path)

# Visualization 2: Missing Values
missing_values_series = df.isnull().sum()
if missing_values_series.sum() > 0:
    plt.figure(figsize=(8, 6))
    missing_values_series[missing_values_series > 0].plot(kind="bar", color="skyblue")
    plt.title("Missing Values per Column")
    missing_values_path = output_dir / "missing_values.png"
    plt.savefig(missing_values_path)

# Visualization 3: Distribution of First Numerical Column
if not numeric_columns.empty:
    plt.figure(figsize=(8, 6))
    sns.histplot(numeric_columns.iloc[:, 0].dropna(), kde=True, color="purple")
    plt.title(f"Distribution of {numeric_columns.columns[0]}")
    dist_path = output_dir / "distribution.png"
    plt.savefig(dist_path)
