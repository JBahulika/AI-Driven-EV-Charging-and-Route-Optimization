
# 🔋 AI-Driven EV Charging & Route Optimization Dashboard

## 🚀 Project Overview
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jbahulika-ai-driven-ev-charging-and-route-optimizati-app-vgzb9h.streamlit.app/)

**🌟 [Click Here view EV dashboard](https://jbahulika-ai-driven-ev-charging-and-route-optimizati-app-vgzb9h.streamlit.app/)**

With the rapid adoption of **Electric Vehicles (EVs)**, challenges such as **charging congestion**, **range anxiety**, and **grid imbalance** are becoming increasingly significant.

This project is an end-to-end AI ecosystem designed to optimize EV infrastructure. It combines predictive machine learning, reinforcement learning, and LLM-driven chat into a single, interactive **Streamlit Dashboard**. The system provides intelligent tools for both EV drivers and station operators.

---

## 📅 Project Milestones

### ✅ WEEK 1: Conceptualization & Data Acquisition
**Goal:** Define problem statements and acquire raw data.
* **Problem Identified:** Drivers face long wait times and unpredictable energy costs. Grid operators face unmanaged load spikes.
* **Objective Defined:** Build a multi-agent AI system to predict demand, optimize routes, and manage pricing.
* **Data Sourced:** Raw datasets for Indian charging stations, EV trip logs, and state-wise EV density.

### ✅ WEEK 2: AI Model Development & Data Cleaning
**Goal:** Clean data and train ML/RL models.
* **Data Cleaning Pipeline:** Developed robust scripts to handle missing coordinates, standardize 30+ state name variations, and remove outlier data.
* **Major Improvisations:**
    * **Pivoted Demand Forecasting:** Shifted from time-series to a **Geospatial Demand Pressure** metric (`evs_per_station`) due to data limitations.
    * **Simulated Environment:** Built a custom Python simulation for dynamic pricing since no historical training data existed.
* **AI Models Built:**
    1.  **Energy Predictor (Random Forest):** Predicts trip efficiency (`kWh/km`) with high accuracy.
    2.  **Dynamic Pricing Agent (Q-Learning):** A Reinforcement Learning agent trained on **50,000 simulated days** to learn optimal pricing strategies.

### ✅ WEEK 3: Full-Stack Integration & Dashboard
**Goal:** Build a user-friendly interface to demonstrate the AI.
* **Streamlit Dashboard:** Built a responsive, multi-page web application (`app.py`) to serve as the front end.
* **Context-Aware Chatbot:** Integrated **OpenAI's GPT-4o** with custom tool use. The chatbot has "Shared Memory"—it knows the results of your energy calculations and pricing simulations from other pages to give smart advice.
* **Visual Upgrades:** Implemented **Plotly** gauges and charts for better data storytelling and a "Theme-Safe" UI that works in both Light and Dark modes.

---

## 💻 Live Dashboard Features
The core of this project is the `app.py` Streamlit application, which integrates all the AI models into a multi-page dashboard.

### 1. 🤖 AI Chatbot Assistant
A context-aware AI assistant that acts as a natural language interface for the entire project.
* **Context-Aware Memory:** The chatbot reads calculations from the other pages. You can run a prediction on the "Energy Predictor" page, then go to the chatbot and ask, **"How can I improve that efficiency?"**
* **Tool-Using AI:** Powered by the OpenAI API, the chatbot can intelligently call the project's other AI models (Energy & Pricing) to answer complex questions directly in the chat.

### 2. 🔮 Energy Predictor (For Drivers)
This page helps drivers solve "range anxiety" by predicting energy consumption for a trip.
* **Interactive Gauge:** A speedometer-style gauge shows the predicted efficiency (`kWh/km`) in real-time.
* **Stat Cards:** Translates efficiency into a simple **Estimated Range (km)** and **Trip Cost (₹ per 100km)**.
* **Live AI Analysis:** A dynamic text box explains *why* the efficiency is good or bad, identifying factors like high speed (wind resistance), cold weather (battery chemistry), or regenerative braking.

### 3. 💰 AI Dynamic Pricing (For Operators)
This page demonstrates the AI agent for managing station congestion.
* **Scenario Simulation:** Allows an operator to select the time of day ("Evening") and station traffic ("High").
* **AI Strategy Analysis:** A Plotly bar chart visually displays the AI's "Success Score" for each pricing option (Low, Medium, High).
* **Clear Recommendation:** The AI highlights the **Best Strategy** (e.g., "Recommended: High Price") to maximize revenue and balance grid load, based on its training.

### 4. 📍 Demand & Station Map
A geospatial visualization of India's charging infrastructure.
* **Heatmap Layer:** Shows where charging stations are most concentrated.
* **Demand Pressure:** Each of the 1,500+ station markers is color-coded (Red/Green) based on the "EVs per Station" in that state, instantly identifying high-pressure, underserved areas.

---

## 🛠️ Tech Stack
* **Frontend & App:** Streamlit
* **Data Science:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Reinforcement Learning:** Custom Q-Learning Simulation (Python + NumPy)
* **Visualization:** Plotly (Charts & Gauges), Folium (Maps)
* **AI Chatbot:** OpenAI (GPT-4o)

---

## ⚙️ How to Run
1.  Clone this repository to your local machine.
2.  Install all required libraries from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run app.py
    ```
4.  To use the chatbot, enter your OpenAI API Key in the sidebar.
````
