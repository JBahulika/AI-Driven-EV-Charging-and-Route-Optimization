import streamlit as st
import pandas as pd
import numpy as np
import joblib
import streamlit.components.v1 as components
import os
import json
import plotly.graph_objects as go
from openai import OpenAI

# --- 1. Page Configuration & Styling ---
st.set_page_config(
    page_title="AI EV Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
    <style>
    /* --- CSS RESET: Removed Global Color Override so standard text adapts to Dark/Light mode --- */

    /* --- 1. CARD STYLING (Always White Background, Always Dark Text) --- */
    .st-card {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Force ALL text inside .st-card to be BLACK/DARK GRAY */
    .st-card, .st-card p, .st-card div, .st-card span, .st-card h1, .st-card h2, .st-card h3, .st-card b {
        color: #1f2937 !important; /* Dark Gray */
    }

    .st-card-title {
        font-size: 12px;
        font-weight: 800;
        color: #6b7280 !important; /* Medium Gray */
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .st-card-value {
        font-size: 24px;
        font-weight: 800;
        color: #111827 !important; /* Almost Black */
    }
    
    .st-card-sub {
        font-size: 13px;
        color: #4b5563 !important; /* Gray */
        margin-top: 4px;
    }

    /* --- 2. METRIC CARDS (Energy Page) --- */
    .st-metric-card {
        background-color: #ffffff !important;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        border-bottom: 5px solid #10b981; /* Green Accent */
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 15px;
    }
    
    /* Force text inside metric card to be dark */
    .st-metric-card div, .st-metric-card span, .st-metric-card p, .st-metric-card h2 {
        color: #1f2937 !important;
    }
    
    .st-metric-icon {
        font-size: 32px;
        margin-bottom: 10px;
    }
    .st-metric-number {
        font-size: 36px;
        font-weight: 900;
        color: #111827 !important;
    }
    .st-metric-label {
        font-size: 14px;
        color: #6b7280 !important;
        font-weight: 700;
        text-transform: uppercase;
    }

    /* --- 3. ANALYSIS BOX (Blue Theme) --- */
    .st-analysis-box {
        background-color: #eff6ff !important; /* Very Light Blue */
        border-left: 6px solid #3b82f6;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    /* Force text inside analysis box to be Dark Blue */
    .st-analysis-box div, .st-analysis-box p, .st-analysis-box b, .st-analysis-box span, .st-analysis-box li {
        color: #1e3a8a !important; /* Dark Blue */
    }
    
    .st-analysis-header {
        font-weight: bold;
        font-size: 18px;
        color: #1d4ed8 !important;
        margin-bottom: 15px;
        border-bottom: 2px solid #bfdbfe;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Session State Initialization ---
if "context_memory" not in st.session_state:
    st.session_state.context_memory = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_energy_model():
    try: return joblib.load('energy_model.joblib')
    except FileNotFoundError: return None

@st.cache_resource
def load_pricing_model():
    try: return np.load('pricing_model_q_table.npy')
    except FileNotFoundError: return None

energy_model = load_energy_model()
q_table = load_pricing_model()

# --- 3. Navigation & Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "AI Chatbot Assistant", 
    "Demand & Station Map", 
    "Energy Predictor", 
    "AI Dynamic Pricing"
])

st.sidebar.markdown("---")

# --- REORDERED: Developer Profile First ---
st.sidebar.markdown("### 👨‍💻 Made by J Bahulika")
st.sidebar.markdown("""
<div style="margin-top: 10px;">
    <a href="https://jbahulika.github.io/index.html" target="_blank" style="text-decoration: none; color: inherit; display: block; margin-bottom: 5px;">
        🌐 <b>Portfolio</b>
    </a>
    <a href="https://www.linkedin.com/in/j-bahulika-8b8237207/" target="_blank" style="text-decoration: none; color: inherit; display: block; margin-bottom: 5px;">
        👔 <b>LinkedIn</b>
    </a>
    <a href="https://github.com/JBahulika" target="_blank" style="text-decoration: none; color: inherit; display: block;">
        💻 <b>GitHub</b>
    </a>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# --- REORDERED: API Key Second ---
api_key = st.sidebar.text_input("OpenAI API Key:", type="password")

# --- 4. Helper Functions ---
def get_state_index(time_of_day, occupancy):
    time_map = {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3}
    occ_map = {'Low': 0, 'Medium': 1, 'High': 2}
    if time_of_day not in time_map or occupancy not in occ_map: return 0
    return time_map[time_of_day] * 3 + occ_map[occupancy]

def get_ai_pricing_recommendation(time_of_day, occupancy):
    if q_table is None: return "Error: Model not loaded."
    idx = get_state_index(time_of_day, occupancy)
    actions = q_table[idx]
    # Fallback logic for empty states
    if np.sum(actions) == 0:
        if occupancy == "High": actions = [50, 80, 100] 
        elif occupancy == "Medium": actions = [60, 100, 70]
        else: actions = [100, 80, 40]
    best = np.argmax(actions)
    price_map = {0: "Low (10₹)", 1: "Medium (15₹)", 2: "High (20₹)"}
    return json.dumps({"recommendation": price_map[best], "state_index": int(idx)})

def predict_energy_cost(day, speed, current, voltage, t_max, t_min, time_day):
    if energy_model is None: return "Error: Model not loaded."
    diff = t_max - t_min
    m, a, e = (1 if time_day=='Morning' else 0), (1 if time_day=='Afternoon' else 0), (1 if time_day=='Evening' else 0)
    feats = np.array([[day, speed, current, voltage, t_max, t_min, diff, m, a, e]])
    pred = energy_model.predict(feats)[0]
    return json.dumps({"kwh_per_km": round(pred, 4)})

# --- 5. OpenAI Tools ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ai_pricing_recommendation",
            "description": "Get optimal pricing based on state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_of_day": {"type": "string", "enum": ["Night", "Morning", "Afternoon", "Evening"]},
                    "occupancy": {"type": "string", "enum": ["Low", "Medium", "High"]}
                },
                "required": ["time_of_day", "occupancy"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_energy_cost",
            "description": "Predict energy efficiency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_of_day": {"type": "string", "enum": ["Night", "Morning", "Afternoon", "Evening"]},
                    "day_of_week": {"type": "integer"},
                    "speed_kmh": {"type": "number"},
                    "current_a": {"type": "number"},
                    "voltage_v": {"type": "number"},
                    "temp_batt_max_c": {"type": "number"},
                    "temp_batt_min_c": {"type": "number"}
                },
                "required": ["time_of_day", "day_of_week", "speed_kmh", "current_a", "voltage_v", "temp_batt_max_c", "temp_batt_min_c"]
            }
        }
    }
]

def handle_chat(prompt_text):
    if not api_key:
        st.error("Please enter API Key in sidebar.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt_text})
    client = OpenAI(api_key=api_key)
    
    system_context = "You are a smart EV Assistant. Be concise."
    if "last_energy" in st.session_state.context_memory:
        system_context += f"\nCONTEXT [Energy]: {st.session_state.context_memory['last_energy']}"
    if "last_pricing" in st.session_state.context_memory:
        system_context += f"\nCONTEXT [Pricing]: {st.session_state.context_memory['last_pricing']}"
    
    messages_to_send = [{"role": "system", "content": system_context}] + st.session_state.messages
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages_to_send, tools=tools
        )
        msg = response.choices[0].message
        
        if msg.tool_calls:
            st.session_state.messages.append(msg)
            messages_to_send.append(msg)
            for tool in msg.tool_calls:
                args = json.loads(tool.function.arguments)
                if tool.function.name == "get_ai_pricing_recommendation":
                    res = get_ai_pricing_recommendation(args["time_of_day"], args["occupancy"])
                elif tool.function.name == "predict_energy_cost":
                    res = predict_energy_cost(args["day_of_week"], args["speed_kmh"], args["current_a"], 
                                            args["voltage_v"], args["temp_batt_max_c"], args["temp_batt_min_c"], 
                                            args["time_of_day"])
                
                tool_msg = {"tool_call_id": tool.id, "role": "tool", "name": tool.function.name, "content": res}
                st.session_state.messages.append(tool_msg)
                messages_to_send.append(tool_msg)
            
            final = client.chat.completions.create(model="gpt-4o-mini", messages=messages_to_send)
            st.session_state.messages.append({"role": "assistant", "content": final.choices[0].message.content})
        else:
            st.session_state.messages.append({"role": "assistant", "content": msg.content})
        st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")

# --- PAGE 1: CHATBOT ---
if page == "AI Chatbot Assistant":
    st.title("📊 AI Context Dashboard")
    with st.container():
        st.markdown("### Live System Telemetry")
        col1, col2 = st.columns(2)
        has_energy = "last_energy" in st.session_state.context_memory
        has_price = "last_pricing" in st.session_state.context_memory
        
        with col1:
            if has_energy:
                e_data = st.session_state.context_memory['last_energy']
                eff = e_data.split("Efficiency: ")[1]
                st.markdown(f"""
                <div class='st-card'>
                    <div class='st-card-title'>🔋 Latest Trip Plan</div>
                    <div class='st-card-value'>{eff}</div>
                    <div class='st-card-sub'>{e_data.split(" ->")[0]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='st-card' style='opacity: 0.7;'>
                    <div class='st-card-title'>🔋 Latest Trip Plan</div>
                    <div class='st-card-value' style='font-size: 16px; font-weight:400;'>No calculation yet</div>
                    <div class='st-card-sub'>Go to 'Energy Predictor' to calculate</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            if has_price:
                p_data = st.session_state.context_memory['last_pricing']
                rec = p_data.split("Recommendation: ")[1]
                st.markdown(f"""
                <div class='st-card'>
                    <div class='st-card-title'>💰 Active Pricing Strategy</div>
                    <div class='st-card-value'>{rec}</div>
                    <div class='st-card-sub'>{p_data.split(". AI")[0]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='st-card' style='opacity: 0.7;'>
                    <div class='st-card-title'>💰 Active Pricing Strategy</div>
                    <div class='st-card-value' style='font-size: 16px; font-weight:400;'>No strategy active</div>
                    <div class='st-card-sub'>Go to 'AI Dynamic Pricing' to simulate</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("#### Suggested queries")
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        if st.button("📉 Analyze my efficiency"): handle_chat("Is my calculated energy efficiency good or bad? How can I improve it?")
    with col_q2:
        if st.button("💸 Explain pricing logic"): handle_chat("Why did you recommend that specific price for this scenario?")
    with col_q3:
        if st.button("🔄 Integrated Analysis"): handle_chat("Based on my last calculated trip and the current station pricing, would it be expensive for me to charge right now?")

    st.markdown("---")
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Type your own question here..."):
        handle_chat(prompt)

# --- PAGE 2: MAP ---
elif page == "Demand & Station Map":
    st.title("🗺️ Live Demand Heatmap")
    if os.path.exists('ev_demand_heatmap.html'):
        with open('ev_demand_heatmap.html', 'r', encoding='utf-8') as f:
            components.html(f.read(), height=700)
    else:
        st.error("Map file not found.")

# --- PAGE 3: ENERGY VISUALIZER ---
elif page == "Energy Predictor":
    st.title("Trip Energy Forecaster")
    
    if energy_model:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Trip Settings")
            speed = st.slider("Speed (km/h)", 0, 150, 60)
            temp_max = st.slider("Battery Temp (°C)", 10, 50, 30)
            current = st.slider("Current (A)", -20, 80, 15)
            time_day = st.selectbox("Time", ["Morning", "Afternoon", "Evening", "Night"])
            day, volt, t_min = 5, 350, temp_max-2
        with col2:
            diff = temp_max - t_min
            m, a, e = (1 if time_day=='Morning' else 0), (1 if time_day=='Afternoon' else 0), (1 if time_day=='Evening' else 0)
            feats = np.array([[day, speed, current, volt, temp_max, t_min, diff, m, a, e]])
            pred = energy_model.predict(feats)[0]
            st.session_state.context_memory['last_energy'] = f"Speed: {speed}km/h, Temp: {temp_max}C, Time: {time_day} -> Efficiency: {pred:.4f} kWh/km"
            
            st.subheader("Efficiency Gauge")
            fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=pred, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "kWh per km", 'font': {'size': 24}}, delta={'reference': 0.20, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}}, gauge={'axis': {'range': [0, 0.5]}, 'bar': {'color': "darkblue"}, 'steps': [{'range': [0, 0.15], 'color': "#2ecc71"}, {'range': [0.15, 0.25], 'color': "#f1c40f"}, {'range': [0.25, 0.50], 'color': "#e74c3c"}]}))
            st.plotly_chart(fig, use_container_width=True)
            
            battery_size = 60 
            estimated_range = int(battery_size / pred) if pred > 0 else 0
            trip_cost = int(pred * 100 * 15)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="st-metric-card">
                    <div class="st-metric-icon">🚗</div>
                    <div class="st-metric-label">Est. Range</div>
                    <div class="st-metric-number">{estimated_range} km</div>
                    <div class="result-sub">On Full Battery (60kWh)</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="st-metric-card">
                    <div class="st-metric-icon">💵</div>
                    <div class="st-metric-label">Trip Cost</div>
                    <div class="st-metric-number">₹ {trip_cost}</div>
                    <div class="result-sub">Per 100 km driven</div>
                </div>
                """, unsafe_allow_html=True)
            
            analysis_items = []
            if speed > 100: analysis_items.append(f"🔻 <b>Speed Penalty:</b> Driving at <b>{speed} km/h</b> creates high drag. Slow to 90 km/h to save ~15% range.")
            elif speed < 30: analysis_items.append(f"⚠️ <b>Idle Drain:</b> At <b>{speed} km/h</b>, AC and lights drain more battery per km than the motor.")
            else: analysis_items.append(f"✅ <b>Speed:</b> <b>{speed} km/h</b> is a highly efficient cruising speed.")

            if temp_max < 15: analysis_items.append(f"❄️ <b>Cold Soak:</b> At <b>{temp_max}°C</b>, battery chemistry slows down. Expect 10-20% range loss.")
            elif temp_max > 35: analysis_items.append(f"🔥 <b>Thermal Load:</b> At <b>{temp_max}°C</b>, cooling systems are working hard to protect the battery.")
            else: analysis_items.append(f"✅ <b>Temp:</b> <b>{temp_max}°C</b> is optimal for battery chemistry.")

            if current > 40: analysis_items.append(f"⚡ <b>Heavy Load:</b> <b>{current} A</b> suggests hard acceleration. Drive smoother.")
            elif current < 0: analysis_items.append(f"♻️ <b>Regen Active:</b> You are recovering energy (<b>{current} A</b>)!")

            analysis_html = "".join([f"<div class='analysis-item'>{item}</div>" for item in analysis_items])
            st.markdown(f"""<div class='st-analysis-box'><div class='st-analysis-header'>📉 Live Efficiency Analysis</div>{analysis_html}</div>""", unsafe_allow_html=True)
    else:
        st.error("Model not found.")

# --- PAGE 5: PRICING VISUALIZER ---
elif page == "AI Dynamic Pricing":
    st.title("AI Smart Pricing Agent")
    if q_table is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Station Situation")
            time_in = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
            occ_in = st.select_slider("How full is the station?", options=["Low", "Medium", "High"])
            story = f"It is {time_in} and the station occupancy is {occ_in}."
            if time_in == "Evening" and occ_in == "High": story += " (Peak Rush Hour!)"
            st.markdown(f"<div class='st-card'><b>Scenario:</b><br>{story}</div>", unsafe_allow_html=True)
        with col2:
            idx = get_state_index(time_in, occ_in)
            vals = q_table[idx]
            if np.sum(vals) == 0:
                if occ_in == "High": vals = [50, 80, 100] 
                elif occ_in == "Medium": vals = [60, 100, 70]
                else: vals = [100, 80, 40]
            best = np.argmax(vals)
            labels = ["Low (10₹)", "Medium (15₹)", "High (20₹)"]
            colors = ['#cccccc', '#cccccc', '#cccccc']
            if best == 0: colors[best] = '#2ecc71'
            elif best == 1: colors[best] = '#f39c12'
            elif best == 2: colors[best] = '#e74c3c'
            rec_text = f"Scenario: {time_in} with {occ_in} occupancy. AI Recommendation: {labels[best]}"
            st.session_state.context_memory['last_pricing'] = rec_text
            fig = go.Figure(data=[go.Bar(x=labels, y=vals, marker_color=colors, text=[f"{v:.0f}" for v in vals], textposition='auto')])
            fig.add_annotation(x=labels[best], y=vals[best], text="🏆 BEST STRATEGY", showarrow=True, arrowhead=1, yshift=10)
            fig.update_layout(title="AI Strategy Analysis (Success Score)", height=400)
            st.plotly_chart(fig, use_container_width=True)
            rec_col = colors[best]
            st.markdown(f"""<div style="background-color: {rec_col}; padding: 20px; border-radius: 10px; color: white; text-align: center;"><h2 style='margin:0; color: white; text-shadow: 0px 1px 2px rgba(0,0,0,0.3);'>Recommended: {labels[best]}</h2></div>""", unsafe_allow_html=True)
    else:
        st.error("Model not found.")