"""
Streamlit Web Application for ML Classification Model Comparison
Wine Quality Dataset - Quality Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="ML Classification Model Comparison",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_results():
    """Load model results"""
    try:
        results_df = pd.read_csv('model/model_results.csv', index_col=0)
        return results_df
    except FileNotFoundError:
        st.error("Model results not found. Please run train_models.py first.")
        return None

@st.cache_resource
def load_models():
    """Load all trained models"""
    model_names = [
        'logistic_regression_model.pkl',
        'decision_tree_model.pkl',
        'k_nearest_neighbor_model.pkl',
        'naive_bayes_model.pkl',
        'random_forest_model.pkl',
        'xgboost_model.pkl'
    ]
    
    models = {}
    for model_name in model_names:
        try:
            with open(f'model/{model_name}', 'rb') as f:
                models[model_name.replace('_model.pkl', '').replace('_', ' ').title()] = pickle.load(f)
        except FileNotFoundError:
            continue
    
    return models

def create_comparison_chart(results_df, metric):
    """Create bar chart for metric comparison"""
    fig = px.bar(
        x=results_df.index,
        y=results_df[metric],
        labels={'x': 'Model', 'y': metric},
        title=f'{metric} Comparison Across Models',
        color=results_df[metric],
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False
    )
    return fig

def create_radar_chart(results_df):
    """Create radar chart for all metrics"""
    fig = go.Figure()
    
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']
    
    for model in results_df.index:
        fig.add_trace(go.Scatterpolar(
            r=results_df.loc[model, metrics].values,
            theta=metrics,
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Radar Chart",
        height=600
    )
    
    return fig

def create_heatmap(results_df):
    """Create heatmap of all metrics"""
    fig = px.imshow(
        results_df.T,
        labels=dict(x="Model", y="Metric", color="Score"),
        x=results_df.index,
        y=results_df.columns,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Model Performance Heatmap"
    )
    fig.update_layout(height=500)
    return fig

def main():
    st.title("🍷 Machine Learning Classification Model Comparison")
    st.markdown("### Wine Quality Prediction - UCI Dataset")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Model Comparison", "Model Details", "Predictions"]
    )
    
    # Load results
    results_df = load_results()
    
    if results_df is None:
        st.warning("⚠️ Please run `python model/train_models.py` first to train the models.")
        return
    
    if page == "Overview":
        st.header("📊 Project Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Models", "6")
        with col2:
            st.metric("Dataset Features", "13")
        with col3:
            st.metric("Dataset Instances", "6,497")
        
        st.markdown("---")
        
        st.subheader("Dataset Description")
        st.markdown("""
        **Dataset:** Wine Quality Dataset  
        **Source:** UCI Machine Learning Repository  
        **Type:** Binary Classification (Wine Quality: Good/Poor)
        
        **Features:** 13 features including:
        - **Acidity features:** fixed acidity, volatile acidity, citric acid, pH
        - **Sugar & density:** residual sugar, density
        - **Salts:** chlorides, free sulfur dioxide, total sulfur dioxide
        - **Other properties:** sulphates, alcohol percentage
        - **Wine type:** red or white wine
        
        **Target Variable:** Binary (0: Poor Quality [<6], 1: Good Quality [≥6])
        
        **Instances:** 6,497 (1,599 red + 4,898 white wines)
        
        **Application:** Predicting wine quality helps winemakers optimize production processes
        and ensure consistent quality based on objective physicochemical tests.
        """)
        
        st.markdown("---")
        
        st.subheader("Models Implemented")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Traditional Models:**
            1. 📈 Logistic Regression
            2. 🌳 Decision Tree Classifier
            3. 🎯 K-Nearest Neighbor Classifier
            """)
        
        with col2:
            st.markdown("""
            **Advanced Models:**
            4. 📊 Gaussian Naive Bayes
            5. 🌲 Random Forest (Ensemble)
            6. 🚀 XGBoost (Ensemble)
            """)
    
    elif page == "Model Comparison":
        st.header("📈 Model Performance Comparison")
        
        # Display results table
        st.subheader("Performance Metrics Table")
        st.dataframe(
            results_df.style.highlight_max(axis=0, color='lightgreen')
                           .format("{:.4f}"),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Metric selector
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_metric = st.selectbox(
                "Select Metric to Compare",
                ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            )
        
        # Bar chart
        st.plotly_chart(
            create_comparison_chart(results_df, selected_metric),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Visualization selector
        viz_type = st.radio(
            "Select Visualization Type",
            ["Radar Chart", "Heatmap"],
            horizontal=True
        )
        
        if viz_type == "Radar Chart":
            st.plotly_chart(create_radar_chart(results_df), use_container_width=True)
        else:
            st.plotly_chart(create_heatmap(results_df), use_container_width=True)
    
    elif page == "Model Details":
        st.header("🔍 Detailed Model Analysis")
        
        selected_model = st.selectbox(
            "Select Model",
            results_df.index.tolist()
        )
        
        st.subheader(f"Performance Metrics: {selected_model}")
        
        # Display metrics in columns
        metrics = results_df.loc[selected_model]
        
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        with col2:
            st.metric("AUC Score", f"{metrics['AUC']:.4f}")
        with col3:
            st.metric("Precision", f"{metrics['Precision']:.4f}")
        with col4:
            st.metric("Recall", f"{metrics['Recall']:.4f}")
        with col5:
            st.metric("F1 Score", f"{metrics['F1']:.4f}")
        with col6:
            st.metric("MCC Score", f"{metrics['MCC']:.4f}")
        
        st.markdown("---")
        
        # Model observations
        st.subheader("Model Observations")
        
        observations = {
            'Logistic Regression': """
            Logistic Regression provides a good baseline performance with interpretable results. 
            It works well for linearly separable data and provides probabilistic predictions. 
            The model shows balanced precision and recall, making it suitable for clinical applications.
            """,
            'Decision Tree': """
            Decision Tree offers high interpretability with clear decision rules. However, it may 
            suffer from overfitting if not properly pruned. The model captures non-linear relationships 
            well but can be sensitive to small variations in the data.
            """,
            'K-Nearest Neighbor': """
            KNN provides good performance for this dataset with moderate computational cost during prediction. 
            The model is sensitive to feature scaling and performs well when similar patients have similar outcomes. 
            However, it can be slower for large datasets.
            """,
            'Naive Bayes': """
            Gaussian Naive Bayes is fast and efficient, especially for smaller datasets. 
            Despite the independence assumption between features, it performs reasonably well. 
            It's particularly useful when training data is limited.
            """,
            'Random Forest': """
            Random Forest ensemble provides robust performance with reduced overfitting compared to single 
            decision trees. It handles feature interactions well and provides feature importance rankings. 
            Generally achieves high accuracy with good generalization.
            """,
            'XGBoost': """
            XGBoost typically achieves the best performance among all models due to its gradient boosting 
            approach and regularization. It handles imbalanced data well and provides excellent predictive 
            power. Requires careful hyperparameter tuning for optimal results.
            """
        }
        
        st.info(observations.get(selected_model, "No observation available for this model."))
        
        # Comparison with best model
        best_model = results_df['F1'].idxmax()
        if selected_model != best_model:
            st.markdown("---")
            st.subheader(f"Comparison with Best Model ({best_model})")
            
            comparison_df = results_df.loc[[selected_model, best_model]]
            st.dataframe(
                comparison_df.style.highlight_max(axis=0, color='lightgreen')
                                  .format("{:.4f}"),
                use_container_width=True
            )
    
    elif page == "Predictions":
        st.header("🎯 Make Predictions")
        
        st.info("💡 This section allows you to input patient data and get predictions from all models.")
        
        # Load models
        models = load_models()
        
        if not models:
            st.error("Models not found. Please train the models first.")
            return
        
        # Input form
        st.subheader("Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 20, 80, 50)
            sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
            trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
            chol = st.slider("Cholesterol", 100, 400, 200)
        
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            restecg = st.selectbox("Resting ECG", [0, 1, 2])
            thalach = st.slider("Max Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
        
        with col3:
            slope = st.selectbox("Slope", [0, 1, 2])
            ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
        
        # Feature engineering
        age_chol_interaction = age * chol
        bp_heart_rate_ratio = trestbps / (thalach + 1)
        age_squared = age ** 2
        
        if st.button("Predict", type="primary"):
            # Prepare input
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                                    exang, oldpeak, slope, ca, thal, age_chol_interaction,
                                    bp_heart_rate_ratio, age_squared]])
            
            # Load scaler
            try:
                with open('model/scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                input_scaled = scaler.transform(input_data)
            except:
                st.error("Scaler not found. Please train the models first.")
                return
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Make predictions
            predictions = {}
            for model_name, model in models.items():
                try:
                    if 'Logistic' in model_name or 'Nearest' in model_name:
                        pred = model.predict(input_scaled)[0]
                        proba = model.predict_proba(input_scaled)[0]
                    else:
                        pred = model.predict(input_data)[0]
                        proba = model.predict_proba(input_data)[0]
                    
                    predictions[model_name] = {
                        'Prediction': 'Disease' if pred == 1 else 'No Disease',
                        'Probability': proba[1]
                    }
                except Exception as e:
                    st.error(f"Error with {model_name}: {str(e)}")
            
            # Display predictions
            pred_df = pd.DataFrame(predictions).T
            
            st.dataframe(
                pred_df.style.format({'Probability': '{:.2%}'}),
                use_container_width=True
            )
            
            # Consensus prediction
            disease_count = sum([1 for p in predictions.values() if p['Prediction'] == 'Disease'])
            
            st.markdown("---")
            st.subheader("Consensus Prediction")
            
            if disease_count >= 4:
                st.error(f"⚠️ **MAJORITY PREDICTION: Heart Disease Detected** ({disease_count}/6 models)")
            elif disease_count >= 3:
                st.warning(f"⚠️ **SPLIT PREDICTION: Inconclusive** ({disease_count}/6 models predict disease)")
            else:
                st.success(f"✅ **MAJORITY PREDICTION: No Heart Disease** ({6-disease_count}/6 models)")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>Machine Learning Classification Assignment | Heart Disease Prediction</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
