# 📊 Time Series Analysis Web App

## 🚀 Overview
This project is a **Time Series Analysis Web App** built using **Streamlit** and containerized with **Docker**. It compares the results of **five different models** and identifies the best-performing model based on evaluation metrics.

## 🛠️ Features
✅ **Compare 5 different time series models**
✅ **Identify the best model based on performance**
✅ **Interactive visualization of results**
✅ **User-friendly Streamlit interface**
✅ **Dockerized for easy deployment and portability**

## 📦 Installation & Setup
Follow these steps to build and run the application using Docker.

### 1️⃣ Clone the Repository
```bash
 git clone https://github.com/Rohand19/ml-docker-app.git
 cd ml-docker-app
```

### 2️⃣ Build the Docker Image
```bash
 docker build -t ml_model .
```

### 3️⃣ Run the Application
```bash
 docker run -p 8501:8501 ml_model
```

## 🖥️ Usage
Once the container is running, open your browser and navigate to:
```
http://localhost:8501
```
Interact with the web app to visualize and analyze time series predictions.

## 📊 Model Evaluation
The app compares five different models, evaluating their performance based on key metrics such as:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared Score (R²)**

The best-performing model is selected and highlighted in the results.

## 🏗️ Tech Stack
- **Python** 🐍
- **Streamlit** 🎨 (Web Interface)
- **Docker** 🐳 (Containerization)
- **Pandas, NumPy, Scikit-Learn, Statsmodels** 📊 (ML Libraries)

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request.

## 📜 License
This project is licensed under the MIT License.

---
🚀 **Developed with ❤️ by Rohan Divakar**

