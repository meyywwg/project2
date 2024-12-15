# Analysis Report

## Story

### Story Summary of the Media Dataset

The dataset under analysis comprises 2,652 entries related to various media, including movies, with several features capturing essential attributes. The primary columns include the date of release, language, type of media, title, author or contributor, and ratings on overall quality and repeatability. 

#### Key Findings

1. **Date and Language Diversity**:
   - The dataset spans a wide range of dates, with 2,055 unique entries, indicating a rich temporal diversity in the media catalog. 
   - The predominant language is English, which appears 1,306 times, suggesting that English-language media dominates this dataset. Other languages represented include 10 additional options, reflecting a multilingual scope.

2. **Types of Media**:
   - The dataset categorizes media into eight distinct types, with "movie" being the most frequent, appearing 2,211 times. This indicates a strong focus on film-related content within the dataset.

3. **Rating Distribution**:
   - The average ratings for overall quality and repeatability are 3.05 and 1.49, respectively, on a scale that has a minimum of 1 and a maximum of 5 for overall quality. The quality ratings have a high standard deviation (0.76), indicating variability in how content is perceived.
   - The median quality rating is 3, suggesting that most media is perceived as average in quality. However, the presence of ratings up to 5 indicates that there is also a significant portion of high-quality content.

4. **Missing Values**:
   - There are notable missing values in the "date" (99 missing values) and "by" (262 missing values) columns. This could impact analyses that depend on understanding media release timelines or attributing content to specific contributors.

5. **Correlation Insights**:
   - The correlation matrix shows a strong positive correlation (0.83) between overall ratings and quality, implying that as the overall rating increases, the perceived quality also rises. There is a moderate correlation (0.51) between overall ratings and repeatability, suggesting that higher-rated media tends to be more repeatable or rewatchable.

#### Actionable Insights

1. **Enhancing Completeness**:
   - Addressing missing values, particularly in the "date" and "by" columns, is crucial for improving the dataset's robustness. This can be achieved through data cleaning processes or acquiring additional data sources to fill in these gaps

## Visualizations
![Correlation Heatmap](correlation_heatmap.png)

![Missing Values](missing_values.png)

![Distribution](distribution.png)
