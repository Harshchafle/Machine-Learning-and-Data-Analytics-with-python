# Google Search Trend Analyzer

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Pytrends](https://img.shields.io/badge/Library-Pytrends-red.svg)
![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

This project provides a powerful and interactive way to analyze Google Search trends using the unofficial Python library **`pytrends`**. It's a comprehensive tool for anyone interested in market research, competitive analysis, trend forecasting, or keyword research. The script allows you to visualize search interest for single or multiple keywords over time and by geographic region.

## ✨ Key Features

- **Single Keyword Analysis:** Dive deep into the search interest for a single keyword.
- **Geographic Insights:** Visualize search interest across different countries on a bar chart and a beautiful interactive world map.
- **Time-Series Analysis:** Track the popularity of a keyword over the last 12 months with a clear line plot.
- **Keyword Comparison:** Compare the search interest of multiple keywords side-by-side to identify competitive trends.
- **Data-Driven Visualizations:** Utilizes `matplotlib` and `plotly` to create static and interactive visualizations that are easy to understand.

## 🛠️ Technologies Used

- **Python 3.11+**
- **Pytrends**: The core library for fetching data from Google Trends.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For creating static data visualizations (bar plots and line charts).
- **Plotly Express**: For generating the interactive choropleth world map.

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.11 or higher installed on your system.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pytrends pandas matplotlib seaborn plotly
    ```

    _Note: The `pip install` output in the notebook confirms that all necessary libraries and their dependencies are already satisfied in your environment. However, this is the command for users to install them locally._

### How to Run

1.  Open the Jupyter Notebook file (`your_notebook_name.ipynb`) in your preferred environment (e.g., Jupyter, VS Code with the Python extension, or Google Colab).
2.  Run the cells sequentially. The script will prompt you to enter a keyword for analysis.
3.  For the multiple keyword comparison, you can modify the `kw_list` variable in the notebook to include the keywords you want to analyze.

## 📊 Visualizations

This project generates the following visualizations:

### 1. Top Countries Searching for a Keyword
A bar chart showing the top 20 countries with the highest search interest for the specified keyword.

![Bar Chart of Top Countries](link_to_your_bar_chart_image.png)

### 2. Search Interest on an Interactive World Map
An interactive choropleth map that visually represents search interest across the globe. You can hover over countries to see their specific interest score.

![Interactive World Map](link_to_your_world_map_image.png)

### 3. Search Interest Over Time
A line graph that plots the search interest of a keyword over the last 12 months.

![Time-Series Plot](link_to_your_time_series_image.png)

### 4. Keyword Comparison Over Time
A comparative line graph that tracks the search interest of multiple keywords, making it easy to identify which one is more popular at any given time.

![Keyword Comparison](link_to_your_comparison_chart_image.png)

## 🤝 Contribution

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have suggestions for improving this project, please fork the repository and create a pull request. You can also open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📞 Contact

Harsh Chafle - chafle2102harsh@gmail.com.com
