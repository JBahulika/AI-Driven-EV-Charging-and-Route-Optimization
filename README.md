# 🔋 AI-Driven EV Charging and Route Optimization System

## 🚀 Project Overview
With the rapid adoption of **Electric Vehicles (EVs)**, challenges such as **charging congestion**, **range anxiety**, and **grid imbalance** are becoming increasingly significant.

This project is an **AI-powered system** designed to optimize EV infrastructure. It predicts energy consumption for specific trips, visualizes real-time demand pressure across India, and uses Reinforcement Learning to dynamically price charging sessions to balance grid load.

---

## 📅 Milestones

### ✅ WEEK 1: Conceptualization & Data Acquisition
**Goal:** Define problem statements and acquire raw data.
* **Problem Identified:** Drivers face long wait times and unpredictable energy costs. Grid operators face unmanaged load spikes.
* **Objective Defined:** Build a multi-agent AI system to predict demand, optimize routes, and manage pricing.
* **Data Sourced:** Raw datasets for Indian charging stations, EV trip logs, and state-wise EV density.

### ✅ WEEK 2: AI Model Development & Data Cleaning
**Goal:** Clean data and train ML/RL models.
* **Data Cleaning Pipeline:** Developed robust cleaning scripts to handle missing coordinates, standardizing 30+ state name variations, and removing impossible outliers (e.g., 900 km/h speed logs) from trip data.
* **Major Improvisations & Pivots:**
    * *Demand Forecasting:* Pivoted from simple time-series forecasting (due to data limitations) to a more impactful **Geospatial Demand Pressure** metric, merging state-level EV density with station locations.
    * *Route Optimization:* Focused on building the "AI Brain" first—a **Random Forest** model that predicts the *energy cost* (`kWh/km`) of a route based on speed, temperature, and time.
* **AI Models Built:**
    1.  **Energy Predictor (Random Forest):** Predicts trip efficiency with an MAE of ~0.04 kWh/km.
    2.  **Dynamic Pricing Agent (Q-Learning):** A Reinforcement Learning agent trained on **50,000 simulated days** to learn optimal pricing strategies (e.g., maximizing revenue during peak demand without alienating customers at night).
* **Final Deliverable:** Three distinct, trained AI components: a geospatial demand map (`.html`), a predictive energy model (`.joblib`), and a dynamic pricing agent (`.npy`).

---

## 🧠 Core AI Components (Week 2 Deliverables)

| Component | Type | Model/Tech | Status |
| :--- | :--- | :--- | :--- |
| **Demand Heatmap** | Visualization | Folium + Geospatial Data Merging | ✅ Complete |
| **Energy Predictor** | Machine Learning | Random Forest Regressor (`sklearn`) | ✅ Trained & Saved |
| **Dynamic Pricing** | Reinforcement Learning | Q-Learning Agent (Custom Simulation) | ✅ Trained & Saved |

---

## 💻 Project Outputs

This project produced three core assets, which are saved in the repository:

1.  **`ev_demand_heatmap.html`**: An interactive Folium map visualizing 1,500+ charging stations. The markers are color-coded (Green/Orange/Red) based on the "EVs per Station" demand pressure in that state.
2.  **`energy_model.joblib`**: A trained `scikit-learn` model that predicts a trip's energy efficiency (kWh/km) based on inputs like speed, battery temperature, and time of day.
3.  **`pricing_model_q_table.npy`**: A trained Q-Table (a NumPy file) that acts as the "brain" for a dynamic pricing agent. It can recommend the optimal price (Low, Medium, High) to maximize revenue based on the time of day and station occupancy.

---

## 🛠️ Tech Stack
* **Python**: Core programming language.
* **Pandas & NumPy**: Advanced data manipulation and simulation.
* **Scikit-Learn**: Training the Random Forest energy model.
* **Folium**: Geospatial data visualization.
