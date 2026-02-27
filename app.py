import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# FORECLOSED PROPERTY – MINIMUM BID PRICE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

# ── Configure Page ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Property Bid Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS for Better UI ─────────────────────────────────────────────────
st.markdown("""
    <style>
        .main {
            padding-top: 1rem;
        }
        .stTitle {
            text-align: center;
            color: #1f77b4;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .prediction-box {
            background-color: #f0f8ff;
            border-left: 5px solid #2ecc71;
            padding: 20px;
            border-radius: 5px;
            margin-top: 2rem;
            text-align: center;
        }
        .prediction-label {
            font-size: 1.2rem;
            color: #333;
            font-weight: bold;
        }
        .prediction-value {
            font-size: 2.5rem;
            color: #2ecc71;
            font-weight: bold;
            margin-top: 0.5rem;
        }
        .input-section {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ── Load Model & Preprocessing Objects ────────────────────────────────────────
@st.cache_resource
def load_model_and_preprocessing():
    """Load trained model, scaler, and encoder from pickle files."""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoder_info = joblib.load('encoder.pkl')
        return model, scaler, encoder_info
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Ensure model.pkl, scaler.pkl, and encoder.pkl are in the app directory.")
        st.stop()

# ── Load Resources ───────────────────────────────────────────────────────────
model, scaler, encoder_info = load_model_and_preprocessing()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# Title
st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>
        🏠Foreclosed Property – Minimum Bid Price Predictor
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
    <p style='text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem;'>
        Predict the minimum bid price of a foreclosed property based on its attributes
    </p>
""", unsafe_allow_html=True)

# ── Input Section ────────────────────────────────────────────────────────────
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    lot_area = st.number_input(
        label="Lot Area (sqm)",
        min_value=1.0,
        max_value=10000.0,
        value=100.0,
        step=10.0,
        help="Enter the total lot area in square meters"
    )

with col2:
    floor_area = st.number_input(
        label="Floor Area (sqm)",
        min_value=0.0,
        max_value=10000.0,
        value=50.0,
        step=10.0,
        help="Enter the total floor area in square meters"
    )

st.divider()

col3, col4, col5 = st.columns(3)

with col3:
    region = st.selectbox(
        label="Region",
        options=["METRO MANILA", "NON-METRO MANILA"],
        help="Select the property region"
    )

with col4:
    remarks = st.selectbox(
        label="Remarks",
        options=["Unoccupied", "Occupied"],
        help="Select property occupancy status"
    )

with col5:
    status_tct = st.selectbox(
        label="Status of TCT",
        options=["TCT under the Bank", "For Title Consolidation"],
        help="Select the title certificate status"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

# Predict Button
if st.button("Predict Minimum Bid Price", use_container_width=True, type="primary"):
    
    try:
        # ── Build Feature Dictionary ────────────────────────────────────────────
        log_lot_area = np.log1p(lot_area)
        log_floor_area = np.log1p(floor_area)
        
        features_dict = {}
        
        # Add base features (as log-transformed)
        features_dict['LOT AREA (sqm)'] = log_lot_area
        features_dict['FLOOR AREA (sqm)'] = log_floor_area
        
        # Engineer new features
        features_dict['PRICE_PER_SQM'] = np.log1p(0 / (lot_area + 1))
        features_dict['TOTAL_AREA'] = log_lot_area + log_floor_area
        features_dict['FLOOR_TO_LOT_RATIO'] = np.log1p(floor_area / (lot_area + 1))
        features_dict['LOG_LOT_x_FLOOR'] = log_lot_area * log_floor_area
        
        # Large property flag (top 25%)
        large_property_threshold = encoder_info.get('large_property_threshold', 5.0)
        features_dict['IS_LARGE_PROPERTY'] = int(log_lot_area >= large_property_threshold)
        
        # ── One-Hot Encoding for Categorical Variables ──────────────────────────
        # Region encoding
        for col in encoder_info['region_cols']:
            if col == f'REGION_{region}':
                features_dict[col] = 1
            else:
                features_dict[col] = 0
        
        # Remarks encoding
        for col in encoder_info['remarks_cols']:
            if col == f'REMARKS_{remarks}':
                features_dict[col] = 1
            else:
                features_dict[col] = 0
        
        # TCT Status encoding
        for col in encoder_info['status_cols']:
            if col == f'STATUS OF TCT_{status_tct}':
                features_dict[col] = 1
            else:
                features_dict[col] = 0
        
        # ── Create DataFrame with Correct Feature Order ──────────────────────────
        X_input = pd.DataFrame([features_dict])
        
        # Ensure all expected columns are present in correct order
        expected_cols = encoder_info['feature_columns']
        X_input = X_input.reindex(columns=expected_cols, fill_value=0)
        
        # ── Apply Scaling ────────────────────────────────────────────────────────
        X_scaled = scaler.transform(X_input)
        
        # ── Make Prediction ──────────────────────────────────────────────────────
        log_prediction = model.predict(X_scaled)[0]
        
        # ── Convert Back from Log Scale ──────────────────────────────────────────
        predicted_price = np.expm1(log_prediction)
        
        # ── Display Result ───────────────────────────────────────────────────────
        st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-label">Estimated Minimum Bid Price</div>
                <div class="prediction-value">₱ {predicted_price:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # ── Display Input Summary ────────────────────────────────────────────────
        with st.expander("Input Summary", expanded=False):
            summary_df = pd.DataFrame({
                'Property Attribute': [
                    'Lot Area',
                    'Floor Area',
                    'Region',
                    'Remarks',
                    'TCT Status'
                ],
                'Value': [
                    f'{lot_area:,.1f} sqm',
                    f'{floor_area:,.1f} sqm',
                    region,
                    remarks,
                    status_tct
                ]
            })
            st.table(summary_df)
        
        # Success message
        st.success("Prediction completed successfully!")
        
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.info("Please check that all inputs are valid and model files are properly loaded.")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()