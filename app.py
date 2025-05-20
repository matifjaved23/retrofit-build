# ML-Based Building Retrofit Recommendation System

# - "Number of Building" column removed
# - Target strategies normalized & canonicalized
# - Targeted improvements for small/imbalanced dataset
# - Specialized optimization for top-3 accuracy

import re
import pandas as pd
import numpy as np
import shap
import streamlit as st
import matplotlib.pyplot as plt
import os
import pickle
import joblib
from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# -------------------------------------------------------------------
# 0. Logger Function
# -------------------------------------------------------------------
def log_info(msg):
    """Print info with clear formatting for easier debugging"""
    print(f"\n{'='*80}\n{msg}\n{'='*80}")

# -------------------------------------------------------------------
# 1. Normalization helpers
# -------------------------------------------------------------------
def normalize_string(s: str) -> str:
    """Lowercase, strip whitespace/newlines/quotes, remove stray punctuation, collapse spaces."""
    s = s.strip().lower()
    s = re.sub(r'[\r\n"]+', ' ', s)
    s = re.sub(r'[^a-z0-9\s\/\-]+', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def canonicalize_strategy(label: str, mapping: dict) -> str:
    """Map normalized label to a canonical human-readable strategy name."""
    norm = normalize_string(label)
    if norm in mapping:
        return mapping[norm]
    return norm.title()

def normalize_dropdown_options(series: pd.Series) -> list:
    """Standardize dropdown values to Title Case, strip, dedupe, sort."""
    opts = series.dropna().astype(str).str.strip().str.title().unique()
    return sorted(opts)

def normalize_column_names(df):
    """Normalize column names to match the expected format"""
    column_mapping = {
        # Area variations
        'Area Footprint (square meters)': 'Area Footprint (sqm)',
        'Area Footprint (m²)': 'Area Footprint (sqm)',
        'Area Footprint': 'Area Footprint (sqm)',
        'Area': 'Area Footprint (sqm)',
        'Building Area': 'Area Footprint (sqm)',
        'Floor Area': 'Area Footprint (sqm)',
        
        # Window variations
        'Window to wall ratio (percent)': 'Window-to-Wall Ratio (%)',
        'Window-to-Wall Ratio': 'Window-to-Wall Ratio (%)',
        'Window to Wall Ratio': 'Window-to-Wall Ratio (%)',
        'WWR': 'Window-to-Wall Ratio (%)',
        'Window Ratio': 'Window-to-Wall Ratio (%)',
        
        # U-value variations
        'Baseline U-value (watts per square meter per Kelvin)': 'Baseline U-value (W/m²·K)',
        'U-value': 'Baseline U-value (W/m²·K)',
        'U value': 'Baseline U-value (W/m²·K)',
        'U-Value': 'Baseline U-value (W/m²·K)',
        'Thermal Transmittance': 'Baseline U-value (W/m²·K)',
        
        # SHGC variations
        'Baseline Solar heat gain coefficient': 'Baseline Solar Heat Gain Coefficient',
        'Solar Heat Gain Coefficient': 'Baseline Solar Heat Gain Coefficient',
        'SHGC': 'Baseline Solar Heat Gain Coefficient',
        'Solar Factor': 'Baseline Solar Heat Gain Coefficient',
        
        # Shading variations
        'Baseline Shading Depth\n(meters)': 'Baseline Shading Depth (m)',
        'Shading Depth': 'Baseline Shading Depth (m)',
        'Shade Depth': 'Baseline Shading Depth (m)',
        'Overhang Depth': 'Baseline Shading Depth (m)',
        
        # Infiltration variations
        'Baseline Infiltrattion rate (Air Changes per Hour)': 'Baseline Infiltration Rate (ACH)',
        'Infiltration Rate': 'Baseline Infiltration Rate (ACH)',
        'Air Changes per Hour': 'Baseline Infiltration Rate (ACH)',
        'ACH': 'Baseline Infiltration Rate (ACH)',
        
        # HVAC variations
        'Baseline HVAC COP Heating': 'Baseline HVAC COP (Heating)',
        'Heating COP': 'Baseline HVAC COP (Heating)',
        'Baseline HVAC COP Cooling': 'Baseline HVAC COP (Cooling)',
        'Cooling COP': 'Baseline HVAC COP (Cooling)',
        
        # Lighting variations
        'Baseline Light\n LPD (watts per square meter per Kelvin)': 'Baseline Lighting LPD (W/m²·K)',
        'Lighting LPD': 'Baseline Lighting LPD (W/m²·K)',
        'LPD': 'Baseline Lighting LPD (W/m²·K)',
        'Lighting Power Density': 'Baseline Lighting LPD (W/m²·K)',
        'Lighting Density': 'Baseline Lighting LPD (W/m²·K)'
    }
    
    # Print original column names for debugging
    print("\nOriginal columns:", df.columns.tolist())
    
    # Map columns
    df.columns = [column_mapping.get(col, col) for col in df.columns]
    
    # Print mapped column names for debugging
    print("Mapped columns:", df.columns.tolist())
    
    return df

# -------------------------------------------------------------------
# 2. Load dataset and define canonical mapping for retrofit strategies
# -------------------------------------------------------------------
# Models and data storage
MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Global variables for models and metadata
models = {}
label_encoders = {}
feature_columns = {}
shap_explainers = {}
feature_importances = {}
class_distributions = {}
component_metrics = {}

# Define feature sets for each component
COMPONENT_FEATURES = {
    'Window': [
        'Climate Zone', 'Building Type', 'Area Footprint (sqm)', 
        'Window-to-Wall Ratio (%)', 'Construction', 
        'Baseline U-value (W/m²·K)', 'Baseline Solar Heat Gain Coefficient'
    ],
    'Wall': [
        'Climate Zone', 'Building Type', 'Area Footprint (sqm)',
        'Construction', 'Baseline U-value (W/m²·K)'
    ],
    'Roof and Floor': [
        'Climate Zone', 'Building Type', 'Area Footprint (sqm)',
        'Construction', 'Baseline U-value (W/m²·K)'
    ],
    'Shade': [
        'Climate Zone', 'Building Type', 'Area Footprint (sqm)',
        'Construction', 'Baseline Shading Depth (m)'
    ],
    'Air Tightness': [
        'Climate Zone', 'Building Type', 'Area Footprint (sqm)',
        'Construction', 'Baseline Infiltration Rate (ACH)'
    ],
    'HVAC': [
        'Climate Zone', 'Building Type', 'Area Footprint (sqm)',
        'Construction', 'Baseline HVAC COP (Heating)', 'Baseline HVAC COP (Cooling)'
    ],
    'Lights': [
        'Climate Zone', 'Building Type', 'Area Footprint (sqm)',
        'Construction', 'Baseline Lighting LPD (W/m²·K)'
    ]
}

# Function to evaluate model w/ focus on top-3 accuracy


def train_models():
    """Train all component models if they don't exist yet"""
    global models, label_encoders, feature_columns, shap_explainers
    global feature_importances, class_distributions, component_metrics
    
    # Check if models already exist
    if os.path.exists(f"{MODELS_DIR}/models.pkl") and os.path.exists(f"{MODELS_DIR}/metadata.pkl"):
        log_info("Loading existing models...")
        models = joblib.load(f"{MODELS_DIR}/models.pkl")
        with open(f"{MODELS_DIR}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            label_encoders = metadata.get("label_encoders", {})
            feature_columns = metadata.get("feature_columns", {})
            shap_explainers = metadata.get("shap_explainers", {})
            feature_importances = metadata.get("feature_importances", {})
            class_distributions = metadata.get("class_distributions", {})
            component_metrics = metadata.get("component_metrics", {})
        return
    
    # If models don't exist, proceed with training
    log_info("Training models...")
    
    
    
    # Initialize containers
    models = {}
    label_encoders = {}
    feature_columns = {}
    shap_explainers = {}
    feature_importances = {}
    class_distributions = {}
    component_metrics = {}
    
    # Save models and metadata
    joblib.dump(models, f"{MODELS_DIR}/models.pkl")
    
    # Save metadata
    metadata = {
        "label_encoders": label_encoders,
        "feature_columns": feature_columns,
        "shap_explainers": shap_explainers,
        "feature_importances": feature_importances,
        "class_distributions": class_distributions,
        "component_metrics": component_metrics
    }
    
    with open(f"{MODELS_DIR}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    log_info("Model training complete!")

# Modified prediction function to handle the "Other Strategies" class
def get_predictions(input_df, component):
    """Get predictions and probabilities for the input data"""
    # Get all probabilities
    probs = models[component].predict_proba(input_df)[0]
    
    # Get class names
    class_names = label_encoders[component].classes_
    
    # Create a list of (class_name, probability) tuples
    predictions = list(zip(class_names, probs))
    
    # Sort by probability (descending)
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Filter out "Other Strategies" completely
    filtered_predictions = [(class_name, prob) for class_name, prob in predictions 
                            if class_name != 'Other Strategies']
    
    # If we have filtered out everything (unlikely), use the original predictions
    if not filtered_predictions:
        filtered_predictions = predictions
    
    # Return top-N predictions
    top_n = min(5, len(filtered_predictions))
    return filtered_predictions[:top_n]

# -------------------------------------------------------------------
# 5. Enhanced Streamlit UI with diagnostics
# -------------------------------------------------------------------
def main():
    # Train or load models at startup
    train_models()
    
    st.set_page_config(
        page_title="Building Retrofit Advisor",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .recommendation-box {
        background-color: #b9b2eb;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1976D2;
        margin-bottom: 1rem;
    }
    .help-text {
        font-size: 0.9rem;
        color: #616161;
        font-style: italic;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header section
    st.markdown('<h1 class="main-header">Building Retrofit Recommendation System</h1>', unsafe_allow_html=True)
    
    # Introduction and explanation
    with st.expander("About this tool", expanded=False):
        st.markdown("""
        This tool helps architects, engineers, and building owners identify optimal energy retrofit 
        strategies based on building characteristics and climate conditions.
        
        **How to use:**
        1. Select the building component you want to retrofit
        2. Enter your building's characteristics and location details
        3. Click "Get Recommendations" to receive optimized retrofit strategies
        
        The AI-powered recommendation engine analyzes your inputs against a database of effective retrofit 
        projects to suggest the most suitable options for your specific situation.
        """)
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["💡 Get Recommendations", "📊 Model Information"])
    
    with tab1:
        st.markdown('<div class="sub-header">Select a building component and enter your building details</div>', unsafe_allow_html=True)
        
        # Component selection with icons and descriptions
        component_info = {
            'Window': {
                'icon': '🪟',
                'description': 'Glazing and window frame improvements to reduce heat transfer and optimize natural light.'
            },
            'Wall': {
                'icon': '🧱',
                'description': 'Wall insulation and cladding upgrades to improve thermal performance.'
            },
            'Roof and Floor': {
                'icon': '🏠',
                'description': 'Enhancements to roof and floor assemblies to reduce heat loss and improve comfort.'
            },
            'Shade': {
                'icon': '☂️',
                'description': 'External shading devices to control solar heat gain and reduce cooling loads.'
            },
            'Air Tightness': {
                'icon': '🌬️',
                'description': 'Improvements to building envelope to reduce air leakage and infiltration.'
            },
            'HVAC': {
                'icon': '❄️',
                'description': 'Heating, ventilation, and air conditioning system upgrades for better energy efficiency.'
            },
            'Lights': {
                'icon': '💡',
                'description': 'Lighting system improvements to reduce energy consumption.'
            }
        }
        
        # Create columns for component selection
        cols = st.columns(3)
        component_options = list(component_info.keys())
        
        # Show component selection as radio buttons with icons
        with st.container():
            st.write("### Select Building Component to Retrofit")
            selected_idx = st.radio(
                "Select Component",
                options=range(len(component_options)),
                format_func=lambda i: f"{component_info[component_options[i]]['icon']} {component_options[i]}",
                horizontal=True,
                label_visibility="collapsed"
            )
            
            component = component_options[selected_idx]
            st.markdown(f"<div class='help-text'>{component_info[component]['description']}</div>", unsafe_allow_html=True)
            st.markdown("---")
        
        # Get the required features for the selected component
        required_features = COMPONENT_FEATURES.get(component, [])
        
        # Create input fields based on component
        inputs = {}
        
        # Common fields for all components
        col1, col2 = st.columns(2)
        with col1:
            # Organize climate zones into categories for better UX
            climate_categories = {
                "Tropical Climates": ['Tropical Rainforest', 'Tropical Monsoon', 'Tropical Savanna', 'Tropical'],
                "Dry Climates": ['Hot Desert', 'Hot Arid', 'Semi Arid', 'Arid'],
                "Moderate (Temperate) Climates": ['Mediterranean', 'Humid Subtropical', 'Temperate Humid', 'Warm Temperate', 
                                                'Temperate Oceanic', 'Temperate Maritime'],
                "Continental Climates": ['Humid Continental', 'Hot Summer and Cold Winter', 'Cold Temperate'],
                "Composite/Mixed": ['Composite']
            }
            
            # First select the category
            category = st.selectbox(
                "Climate Category",
                options=list(climate_categories.keys()),
                key="climate_category"
            )
            
            # Then select the specific climate within the category
            inputs['Climate Zone'] = st.selectbox(
                "Climate Zone", 
                options=climate_categories[category],
                key="climate",
                help="Select the specific climate zone that best matches your location."
            )
        
        with col2:
            # Enhanced building type options with more detail
            building_categories = {
                "Residential": ['Single-Family Home', 'Multi-Family Apartment', 'Condominium', 'Townhouse'],
                "Commercial": ['Office Building', 'Retail Store', 'Shopping Mall', 'Hotel/Hospitality', 'Restaurant'],
                "Institutional": ['Educational/School', 'Healthcare Facility', 'Government Building'],
                "Industrial": ['Manufacturing Facility', 'Warehouse', 'Data Center'],
                "Other": ['Mixed-Use', 'Transportation', 'Recreational']
            }
            
            # First select the building category
            building_category = st.selectbox(
                "Building Category",
                options=list(building_categories.keys()),
                key="building_category"
            )
            
            # Then select the specific building type
            building_type_detailed = st.selectbox(
                "Building Type",
                options=building_categories[building_category],
                key="building_type_detailed",
                help="Select the specific type of building you're working with."
            )
            
            # Map the detailed building type to the general category for model compatibility
            building_type_map = {
                'Single-Family Home': 'Residential', 
                'Multi-Family Apartment': 'Residential',
                'Condominium': 'Residential', 
                'Townhouse': 'Residential',
                'Office Building': 'Commercial', 
                'Retail Store': 'Commercial',
                'Shopping Mall': 'Commercial', 
                'Hotel/Hospitality': 'Commercial',
                'Restaurant': 'Commercial',
                'Educational/School': 'Educational', 
                'Healthcare Facility': 'Healthcare',
                'Government Building': 'Commercial',
                'Manufacturing Facility': 'Industrial', 
                'Warehouse': 'Industrial',
                'Data Center': 'Industrial',
                'Mixed-Use': 'Commercial', 
                'Transportation': 'Commercial',
                'Recreational': 'Commercial'
            }
            
            # Map to general category for model compatibility
            inputs['Building Type'] = building_type_map.get(building_type_detailed, building_category)
        
        col3, col4 = st.columns(2)
        with col3:
            inputs['Area Footprint (sqm)'] = st.number_input(
                "Area Footprint (sqm)",
                min_value=0.0,
                value=100.0,
                key="area"
            )
        
        with col4:
            construction_materials = {
                "Concrete": ['Reinforced Concrete', 'Precast Concrete', 'Concrete Block', 'Standard Concrete'],
                "Steel": ['Steel Frame', 'Steel and Glass', 'Light Gauge Steel'],
                "Wood": ['Timber Frame', 'Light Frame Wood', 'Engineered Wood'],
                "Masonry": ['Brick', 'Stone', 'Concrete Block Masonry'],
                "Mixed/Other": ['Hybrid Construction', 'Mixed Materials', 'Prefabricated Components']
            }
            
            # First select the construction category
            construction_category = st.selectbox(
                "Construction Material Category",
                options=list(construction_materials.keys()),
                key="construction_category",
                help="Main construction material category"
            )
            
            # Then select the specific construction type
            construction_detail = st.selectbox(
                "Construction Type",
                options=construction_materials[construction_category],
                key="construction_detail",
                help="Select the specific construction type and materials"
            )
            
            # Map the detailed construction to the general category for model compatibility
            inputs['Construction'] = construction_category
        
        # Component-specific fields
        if component == 'Window':
            col5, col6 = st.columns(2)
            with col5:
                inputs['Window-to-Wall Ratio (%)'] = st.number_input(
                    "Window-to-Wall Ratio (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=30.0,
                    key="wwr"
                )
            with col6:
                inputs['Baseline U-value (W/m²·K)'] = st.number_input(
                    "Baseline U-value (W/m²·K)",
                    min_value=0.0,
                    value=2.5,
                    key="u_value"
                )
            inputs['Baseline Solar Heat Gain Coefficient'] = st.number_input(
                "Baseline Solar Heat Gain Coefficient",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                key="shgc"
            )
        
        elif component == 'Wall':
            inputs['Baseline U-value (W/m²·K)'] = st.number_input(
                "Baseline U-value (W/m²·K)",
                min_value=0.0,
                value=0.5,
                key="u_value"
            )
        
        elif component == 'Roof and Floor':
            inputs['Baseline U-value (W/m²·K)'] = st.number_input(
                "Baseline U-value (W/m²·K)",
                min_value=0.0,
                value=0.3,
                key="u_value"
            )
        
        elif component == 'Shade':
            inputs['Baseline Shading Depth (m)'] = st.number_input(
                "Baseline Shading Depth (m)",
                min_value=0.0,
                value=0.5,
                key="shade_depth"
            )
        
        elif component == 'Air Tightness':
            inputs['Baseline Infiltration Rate (ACH)'] = st.number_input(
                "Baseline Infiltration Rate (ACH)",
                min_value=0.0,
                value=0.5,
                key="infiltration"
            )
        
        elif component == 'HVAC':
            col5, col6 = st.columns(2)
            with col5:
                inputs['Baseline HVAC COP (Heating)'] = st.number_input(
                    "Baseline HVAC COP (Heating)",
                    min_value=0.0,
                    value=2.5,
                    key="cop_heating"
                )
            with col6:
                inputs['Baseline HVAC COP (Cooling)'] = st.number_input(
                    "Baseline HVAC COP (Cooling)",
                    min_value=0.0,
                    value=2.5,
                    key="cop_cooling"
                )

        elif component == 'Lights':
            inputs['Baseline Lighting LPD (W/m²·K)'] = st.number_input(
                "Baseline Lighting LPD (W/m²·K)",
                min_value=0.0,
                value=10.0,
                key="lpd"
            )

        if st.button("Get Recommendations"):
            if component in models:
                # Create input DataFrame
                input_df = pd.DataFrame([inputs])
                
                # Preprocess input
                input_df = preprocess_input(input_df, component)
                
                # Get predictions
                predictions = get_predictions(input_df, component)
                
                # Display results in an appealing format
                st.markdown('<h2 class="sub-header">Top Recommendations</h2>', unsafe_allow_html=True)
                
                # Results in columns with visual elements
                rec_cols = st.columns(3)
                
                for i, (strategy, prob) in enumerate(predictions[:3], 1):
                    with rec_cols[i-1]:
                        # Create a card-like element for each recommendation
                        confidence_color = "#388E3C" if prob > 0.6 else "#FFA000" if prob > 0.3 else "#F57F17"
                        
                        # Medal emojis for ranking
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" 
                        
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; height: 100%;">
                            <h3 style="text-align: center; margin-bottom: 10px; color: #0D47A1;">{medal} Option {i}</h3>
                            <p style="font-size: 1.1rem; text-align: center; font-weight: bold;">{strategy}</p>
                            <div style="margin: 15px 0;">
                                <div style="background-color: #f0f0f0; border-radius: 10px; height: 10px; width: 100%;">
                                    <div style="background-color: {confidence_color}; width: {prob*100}%; height: 10px; border-radius: 10px;"></div>
                                </div>
                                <p style="text-align: center; margin-top: 5px;">Confidence: {prob:.1%}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Additional recommendations
                if len(predictions) > 3:
                    with st.expander("View Additional Recommendations"):
                        for i, (strategy, prob) in enumerate(predictions[3:], 4):
                            st.markdown(f"""
                            <div style="margin-bottom: 10px; padding: 8px 15px; background-color: #f8f9fa; border-radius: 4px;">
                                <span style="font-weight: bold;">Option {i}:</span> {strategy} (Confidence: {prob:.1%})
                            </div>
                            """, unsafe_allow_html=True)
                
                # Show feature importance if available
                if component in feature_importances:
                    st.markdown('<h2 class="sub-header">Key Factors Influencing Recommendations</h2>', unsafe_allow_html=True)
                    st.markdown("""
                    <div class="help-text">
                    These factors had the greatest influence on the recommendation algorithm's decision.
                    </div>
                    """, unsafe_allow_html=True)
                    plot_feature_importance(component)
                    
                # Recommendation summary and next steps
                st.markdown('<h2 class="sub-header">Implementation Considerations</h2>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="recommendation-box">
                    <p><strong>For {component} retrofit in {inputs['Building Type']} buildings:</strong></p>
                    <ul>
                        <li>Consider local building codes and regulations before implementation</li>
                        <li>Evaluate cost-benefit ratio and potential payback period</li>
                        <li>Consult with specialized contractors for detailed quotations</li>
                        <li>Check for available rebates or incentives in your region</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"No model available for {component}. This might be because:")
                st.write("1. The model hasn't been trained yet")
                st.write("2. There wasn't enough data for this component")
                st.write("3. There was an error during model training")
                
                # If it's HVAC, provide additional context
                if component == "HVAC":
                    st.info("Note: Our HVAC recommendation system uses advanced feature engineering to predict optimal solutions based on your climate zone, building type and COP values. For detailed implementation, consider consulting an HVAC specialist.")
                    
    with tab2:
        st.markdown('<div class="sub-header">About the Recommendation Engine</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Model Architecture
        
        This tool uses advanced machine learning techniques to analyze your building characteristics and predict the most effective 
        retrofit strategies for each component. 
        
        **Key Features:**
        - Specialized algorithms for each building component
        - Adaptation to different climate zones and building types
        - Top-3 focused recommendation accuracy
        - Feature importance analysis for decision transparency
        """)
        
        # Show model performance metrics if available
        if component_metrics:
            st.markdown("### Model Performance Metrics")
            
            metrics_data = []
            for comp, metrics in component_metrics.items():
                metrics_data.append({
                    'Component': comp,
                    'Accuracy': round(metrics.get('accuracy', 0), 2),
                    'Top-3 Accuracy': round(metrics.get('top3_accuracy', 0), 2),
                    'Number of Classes': len(metrics.get('per_class', {}))
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df)
        
        # Dataset information
        st.markdown("### Dataset Information")
        st.markdown("""
        The recommendation engine is trained on a dataset of real-world building retrofit projects, capturing:
        
        - Various climate zones from tropical to continental regions
        - Multiple building types and construction methods
        - Component-specific parameters and their impact on energy efficiency
        - Proven retrofit strategies with successful implementation history
        
        The models are optimized for small and medium-sized datasets using specialized machine learning techniques
        to ensure robust recommendations even with limited data.
        """)
        
        # Disclaimer
        st.markdown("### Disclaimer")
        st.markdown("""
        <div style="font-size:0.9rem; color:#666; padding:10px; border-left:3px solid #ddd;">
        The recommendations provided are based on statistical learning from past projects and general patterns.
        They should be used as a starting point for retrofit planning and should be verified by qualified professionals.
        Local building codes, specific site conditions, and detailed energy modeling may impact the suitability
        of the recommended strategies.
        </div>
        """, unsafe_allow_html=True)

def preprocess_input(input_df, component):
    """Preprocess input data to match the model's expected format"""
    # Get the required features for this component
    required_features = COMPONENT_FEATURES.get(component, [])
    
    # Ensure all required features are present
    for feature in required_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Special preprocessing for HVAC component to enhance predictions
    if component == 'HVAC':
        input_df = enhance_hvac_features(input_df)
    
    # Handle categorical variables
    cat_cols = ['Climate Zone', 'Building Type', 'Construction']
    input_enc = pd.get_dummies(input_df, columns=cat_cols, drop_first=False)
    
    # Align to training features
    required_cols = feature_columns[component]
    for col in required_cols:
        if col not in input_enc:
            input_enc[col] = 0
    
    # Make sure we only use columns that were in training
    input_enc = input_enc.reindex(columns=required_cols, fill_value=0)
    
    return input_enc

def enhance_hvac_features(input_df):
    """Enhance HVAC features by adding derived features and defaults based on climate zone and building type"""
    # Make a deep copy to avoid modifying the original
    df = input_df.copy()
    
    # 1. Add year built - default based on building type
    building_type_age_map = {
        # Residential buildings
        'Residential': 2000,
        'Single-Family Home': 1995,
        'Multi-Family Apartment': 2000,
        'Condominium': 2005,
        'Townhouse': 2000,
        
        # Commercial buildings
        'Commercial': 2005,
        'Office Building': 2008,
        'Retail Store': 2005,
        'Shopping Mall': 2010,
        'Hotel/Hospitality': 2005,
        'Restaurant': 2005,
        
        # Institutional buildings
        'Educational': 1995,
        'Educational/School': 1995,
        'Healthcare': 2010,
        'Healthcare Facility': 2010,
        'Government Building': 2000,
        
        # Industrial buildings
        'Industrial': 2000,
        'Manufacturing Facility': 1995,
        'Warehouse': 2000,
        'Data Center': 2015,
        
        # Other building types
        'Mixed-Use': 2010,
        'Transportation': 2005,
        'Recreational': 2010
    }
    building_type = df['Building Type'].iloc[0]
    df['Year Built'] = building_type_age_map.get(building_type, 2000)
    
    # 2. Add window to wall ratio - default based on building type
    building_type_wwr_map = {
        # Residential buildings
        'Residential': 15.0,
        'Single-Family Home': 15.0,
        'Multi-Family Apartment': 20.0,
        'Condominium': 25.0,
        'Townhouse': 18.0,
        
        # Commercial buildings
        'Commercial': 40.0,
        'Office Building': 45.0,
        'Retail Store': 20.0,
        'Shopping Mall': 25.0,
        'Hotel/Hospitality': 30.0,
        'Restaurant': 25.0,
        
        # Institutional buildings
        'Educational': 30.0,
        'Educational/School': 30.0,
        'Healthcare': 30.0,
        'Healthcare Facility': 35.0,
        'Government Building': 35.0,
        
        # Industrial buildings
        'Industrial': 20.0,
        'Manufacturing Facility': 15.0,
        'Warehouse': 10.0,
        'Data Center': 12.0,
        
        # Other building types
        'Mixed-Use': 35.0,
        'Transportation': 40.0,
        'Recreational': 30.0
    }
    df['Window to wall ratio (percent)'] = building_type_wwr_map.get(building_type, 30.0)
    
    # 3. Add temperature and humidity based on climate zone
    climate_data = {
        # Tropical Climates
        'Tropical Rainforest': {'high_temp': 34.0, 'low_temp': 22.0, 'high_humidity': 95.0, 'low_humidity': 70.0},
        'Tropical Monsoon': {'high_temp': 35.0, 'low_temp': 20.0, 'high_humidity': 90.0, 'low_humidity': 65.0},
        'Tropical Savanna': {'high_temp': 33.0, 'low_temp': 18.0, 'high_humidity': 80.0, 'low_humidity': 40.0},
        'Tropical': {'high_temp': 34.0, 'low_temp': 20.0, 'high_humidity': 90.0, 'low_humidity': 60.0},
        
        # Dry Climates
        'Hot Desert': {'high_temp': 45.0, 'low_temp': 10.0, 'high_humidity': 40.0, 'low_humidity': 10.0},
        'Hot Arid': {'high_temp': 42.0, 'low_temp': 12.0, 'high_humidity': 35.0, 'low_humidity': 12.0},
        'Semi Arid': {'high_temp': 38.0, 'low_temp': 8.0, 'high_humidity': 50.0, 'low_humidity': 20.0},
        'Arid': {'high_temp': 40.0, 'low_temp': 10.0, 'high_humidity': 45.0, 'low_humidity': 15.0},
        
        # Moderate (Temperate) Climates
        'Mediterranean': {'high_temp': 30.0, 'low_temp': 10.0, 'high_humidity': 70.0, 'low_humidity': 40.0},
        'Humid Subtropical': {'high_temp': 35.0, 'low_temp': 5.0, 'high_humidity': 85.0, 'low_humidity': 55.0},
        'Temperate Humid': {'high_temp': 28.0, 'low_temp': 8.0, 'high_humidity': 80.0, 'low_humidity': 50.0},
        'Warm Temperate': {'high_temp': 32.0, 'low_temp': 7.0, 'high_humidity': 75.0, 'low_humidity': 45.0},
        'Temperate Oceanic': {'high_temp': 25.0, 'low_temp': 0.0, 'high_humidity': 85.0, 'low_humidity': 65.0},
        'Temperate Maritime': {'high_temp': 24.0, 'low_temp': 3.0, 'high_humidity': 82.0, 'low_humidity': 60.0},
        
        # Continental Climates
        'Humid Continental': {'high_temp': 30.0, 'low_temp': -10.0, 'high_humidity': 80.0, 'low_humidity': 40.0},
        'Hot Summer and Cold Winter': {'high_temp': 32.0, 'low_temp': -5.0, 'high_humidity': 75.0, 'low_humidity': 35.0},
        'Cold Temperate': {'high_temp': 22.0, 'low_temp': -15.0, 'high_humidity': 70.0, 'low_humidity': 30.0},
        
        # Composite/Mixed
        'Composite': {'high_temp': 34.0, 'low_temp': 0.0, 'high_humidity': 80.0, 'low_humidity': 40.0}
    }
    
    climate_zone = df['Climate Zone'].iloc[0]
    climate_info = climate_data.get(climate_zone, climate_data['Temperate Humid'])  # Default to Temperate Humid if not found
    
    df['Air temperature high (degrees Celsius)'] = climate_info['high_temp']
    df['Air temperature low (degrees Celsius)'] = climate_info['low_temp']
    df['Relative humidity high (percent)'] = climate_info['high_humidity']
    df['Relative humidity low (percent)'] = climate_info['low_humidity']
    
    # 4. Add derived features - efficiency ratios and differences
    df['COP Heating to Cooling Ratio'] = df['Baseline HVAC COP (Heating)'] / df['Baseline HVAC COP (Cooling)'].replace(0, 1)
    df['COP Range'] = df['Baseline HVAC COP (Cooling)'] - df['Baseline HVAC COP (Heating)']
    df['COP Total'] = df['Baseline HVAC COP (Cooling)'] + df['Baseline HVAC COP (Heating)']
    
    # 5. Complexity score based on building size (larger buildings = more complex HVAC needs)
    df['Building Size Category'] = pd.cut(
        df['Area Footprint (sqm)'], 
        bins=[0, 500, 5000, 50000, float('inf')], 
        labels=[1, 2, 3, 4]
    ).astype(float)
    
    return df

def plot_feature_importance(component):
    """Plot feature importance for the selected component"""
    if component not in feature_importances:
        return
    
    # Get top 10 features
    top_features = feature_importances[component].head(10)
    
    # Create horizontal bar chart
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    y_pos = np.arange(len(top_features))
    
    # Clean feature names for display
    clean_names = [name.replace('_', ' ').title() for name in top_features.index]
    
    ax.barh(y_pos, top_features.values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names)
    ax.invert_yaxis()  # Highest values at the top
    ax.set_xlabel('Importance')
    ax.set_title(f'Top Features for {component}')
    plt.tight_layout()
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()