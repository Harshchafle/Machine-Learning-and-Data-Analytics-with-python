# Netflix Movie Data Analysis

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Pandas](https://img.shields.io/badge/Library-Pandas-red.svg)
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-green.svg)
![Seaborn](https://img.shields.io/badge/Library-Seaborn-purple.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Objective

The primary goal of this project is to perform an Exploratory Data Analysis (EDA) on a dataset of Netflix movies. By cleaning, processing, and visualizing the data, this project aims to uncover key trends and insights, such as the most popular genres, the distribution of movie ratings, and release patterns over time.

---

## 📂 Dataset

This analysis uses the **`mymoviedb.csv`** dataset, which contains information on a wide range of movies available on Netflix. The dataset includes columns such as `Release_Date`, `Title`, `Popularity`, `Vote_Count`, `Vote_Average`, and `Genre`.

The dataset was obtained from Kaggle.

---

## 🛠️ Tools & Libraries

- **Python 3.11+**
- **Pandas**: For data manipulation, cleaning, and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For static data visualization.
- **Seaborn**: For creating aesthetically pleasing and informative statistical graphics.

---

## 📝 Analysis & Key Findings

The Jupyter Notebook (`your_notebook_name.ipynb`) walks through the entire data analysis process, from initial data inspection to drawing conclusions. Here are some of the key questions answered by this analysis:

1.  **What is the most frequent genre of movies released?**
    - **Answer:** The most common genre is **Drama**, which appears most frequently in the dataset.
    
2.  **How are the movies distributed across different vote average categories?**
    - **Answer:** The largest category is **`average`**, indicating a high concentration of movies with mid-range ratings.
    
3.  **Which movie has the highest popularity score?**
    - **Answer:** **Spider-Man: No Way Home** is the most popular movie in the dataset.
    
4.  **What year saw the most movies released?**
    - **Answer:** The year **2021** had the highest number of movie releases in this dataset.

---

## 📊 Visualizations

This project includes several visualizations to illustrate the key findings:

| Plot Type | Description |
| :--- | :--- |
| **Bar Plot** | Displays the frequency of each movie genre. |
| **Bar Plot** | Shows the distribution of movies across the `Vote_Average` categories. |
| **Histogram** | Visualizes the distribution of movie releases by year, highlighting the most active years. |



---

## 🚀 How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy seaborn matplotlib
    ```

3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook your_notebook_name.ipynb
    ```

---

## 🤝 Contribution

Feel free to fork this repository, explore the data further, and contribute your own analyses or improvements. If you find any issues or have suggestions, please open an issue.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
