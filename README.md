# NYC-Regents-Disparities

This project analyzes disparities in New York City public high school Regents exam outcomes using a multi-level model across key demographics, including race, gender, and geographic location. By integrating standardized testing data with information on school staffing, class sizes, and student demographics, the project aims to uncover systemic inequities and inform data-driven policy decisions.

---

## 📊 Project Overview

The New York State Regents Exams are standardized assessments required for high school graduation. However, performance disparities have been observed across different student groups. This project seeks to:

- Analyze Regents exam outcomes from 2017 to 2023.
- Identify correlations between exam performance and factors such as race, gender, and school location.
- Assess the impact of class sizes, faculty demographics, and school resources on student achievement.
- Provide visualizations and statistical analyses to support findings.

---

## 🗂️ Repository Structure

```
NYC-Regents-Disparities/
├── data/
│   ├── raw/                 # Original datasets
│   └── cleaned/             # Cleaned and merged datasets
├── notebooks/               # Jupyter notebooks for analysis
├── src/                     # Python scripts for data processing
├── visuals/                 # Generated charts and graphs
├── .github/
│   └── workflows/           # GitHub Actions workflows
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 📁 Data Sources

The project utilizes publicly available datasets from https://opendata.cityofnewyork.us/, including:

- NYC Department of Education Regents exam results
- School demographic and staffing data
- Class size reports
- Geographic information for school locations

All data sources are cited within the respective notebooks and scripts.

---

## 🛠️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vikobaldigi/NYC-Regents-Disparities.git
   cd NYC-Regents-Disparities
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 📈 Usage

- Navigate to the `notebooks/` directory to explore Jupyter notebooks containing data analyses and visualizations.
- Use the scripts in the `src/` directory to preprocess data or generate specific plots.
- Visual outputs are stored in the `visuals/` directory for reference.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a pull request.

Please ensure your code adheres to the project's coding standards and includes appropriate documentation.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For questions or collaborations, please contact [Vi Kobal a.k.a. Philip Kovacevic](mailto:vk@vikobal.digial?subject=[GitHub]%20Source%20Han%20Sans).
