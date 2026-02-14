"""
Streamlit Web Application for ML Classification Model Comparison
Wine Quality Dataset - Quality Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics as mts

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

@st.cache_data
def load_test_data():
    """Load test data for confusion matrix generation"""
    try:
        with open('model/test_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error("Test data not found. Please run train_models.py first.")
        return None

@st.cache_resource
def load_models():
    """Load all trained models"""
    model_mapping = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl',
        'K-Nearest Neighbor': 'k_nearest_neighbor_model.pkl',
        'Naive Bayes': 'naive_bayes_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }
    
    models = {}
    for display_name, file_name in model_mapping.items():
        try:
            with open(f'model/{file_name}', 'rb') as f:
                models[display_name] = pickle.load(f)
        except FileNotFoundError:
            continue
    
    return models

@st.cache_resource
def load_scaler():
    """Load the scaler"""
    try:
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except FileNotFoundError:
        st.error("Scaler not found. Please run train_models.py first.")
        return None

def create_comparison_chart(results_df, metric):
    """Create bar chart for metric comparison using Matplotlib/Seaborn"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create barplot
    sns.barplot(
        x=results_df.index, 
        y=results_df[metric], 
        hue=results_df.index, 
        palette='viridis', 
        ax=ax,
        legend=False 
    )
    
    ax.set_title(f'{metric} Comparison Across Models')
    ax.set_ylabel(metric)
    ax.set_xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def create_radar_chart(results_df):
    """Create radar chart for all metrics using Matplotlib"""
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']
    
    # Number of variables
    N = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], metrics)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)
    
    # Plot each model
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    
    for idx, (model_name, row) in enumerate(results_df.iterrows()):
        values = row[metrics].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Model Performance Radar Chart")
    
    return fig

def create_heatmap(results_df):
    """Create heatmap of all metrics using Seaborn"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(
        results_df, 
        annot=True, 
        cmap='RdYlGn', 
        fmt='.3f', 
        ax=ax,
        vmin=0, 
        vmax=1
    )
    
    ax.set_title("Model Performance Heatmap")
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names=['Poor', 'Good']):
    """Plot confusion matrix"""
    cm = mts.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    return fig

def main():
    st.title("🍷 Machine Learning Classification Model Comparison")
    st.markdown("### Wine Quality Prediction - UCI Dataset")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Model Comparison", "Model Details", "Predictions", "Custom Data Prediction"]
    )
    
    # Load results
    results_df = load_results()
    test_data = load_test_data()
    models = load_models()
    scaler = load_scaler()
    
    if results_df is None or test_data is None or not models or scaler is None:
        st.warning("⚠️ Please run `python model/train_models.py` first to train the models.")
        return
    
    if page == "Overview":
        st.header("📊 Project Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Models", "6")
        with col2:
            st.metric("Dataset Features", "12")
        with col3:
            st.metric("Dataset Instances", "6,497")
        
        st.markdown("---")
        
        st.subheader("Dataset Description")
        st.markdown("""
        **Dataset:** Wine Quality Dataset  
        **Source:** UCI Machine Learning Repository  
        **Type:** Binary Classification (Wine Quality: Good/Poor)
        
        **Features:** 12 features (11 physicochemical + 1 type) including:
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
        st.pyplot(create_comparison_chart(results_df, selected_metric))
        
        st.markdown("---")
        
        # Visualization selector
        viz_type = st.radio(
            "Select Visualization Type",
            ["Heatmap", "Radar Chart"],
            horizontal=True
        )
        
        if viz_type == "Radar Chart":
            st.pyplot(create_radar_chart(results_df))
        else:
            st.pyplot(create_heatmap(results_df))
    
    elif page == "Model Details":
        st.header("🔍 Detailed Model Analysis")
        
        selected_model = st.selectbox(
            "Select Model",
            results_df.index.tolist()
        )

        # Model observations
        st.subheader("Model Observations")
        
        observations = {
            'Logistic Regression': """
            Logistic Regression provides a good baseline performance with interpretable results. 
            It works well for linearly separable data and provides probabilistic predictions. 
            The model shows balanced precision and recall, suitable for classification tasks where
            interpretability is key.
            """,
            'Decision Tree': """
            Decision Tree offers high interpretability with clear decision rules. However, it may 
            suffer from overfitting if not properly pruned. The model captures non-linear relationships 
            well but can be sensitive to small variations in chemical properties.
            """,
            'K-Nearest Neighbor': """
            KNN provides good performance for local neighborhoods of data points. 
            The model is sensitive to feature scaling (physicochemical properties vary in range).
            It performs well when wines with similar properties have similar quality ratings.
            """,
            'Naive Bayes': """
            Gaussian Naive Bayes is fast and efficient. 
            Despite the independence assumption between physicochemical features (which may not always hold true),
            it performs reasonably well and is useful as a baseline.
            """,
            'Random Forest': """
            Random Forest ensemble provides robust performance with reduced overfitting compared to single 
            decision trees. It handles interactions between chemical features well and provides feature importance rankings. 
            Generally achieves high accuracy.
            """,
            'XGBoost': """
            XGBoost typically achieves the best performance among all models due to its gradient boosting 
            approach and regularization. It handles complex non-linear relationships between ingredients
            and quality well.
            """
        }
        
        st.info(observations.get(selected_model, "No observation available for this model."))

        st.markdown("---")
        
        
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
        
        # Confusion Matrix and Classification Report
        st.subheader("Confusion Matrix & Classification Report")
        
        model = models[selected_model]
        
        # Prepare data based on model type
        if 'Logistic' in selected_model or 'Nearest' in selected_model:
            X_eval = test_data['X_test_scaled']
        else:
            X_eval = test_data['X_test']
            
        y_pred = model.predict(X_eval)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Confusion Matrix**")
            st.pyplot(plot_confusion_matrix(test_data['y_test'], y_pred))
            
        with col2:
            st.markdown("**Classification Report**")
            report = mts.classification_report(test_data['y_test'], y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"))
        
    
        # Comparison with best model
        best_model = results_df['F1'].idxmax()
        if selected_model == best_model:
            st.markdown("---")
            st.success(f"✅ **{selected_model} is the best model**")
        else:
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
        st.subheader("Wine Characteristics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.0)
            volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.3)
            citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
            residual_sugar = st.slider("Residual Sugar", 0.5, 66.0, 2.0)
            
        with col2:
            chlorides = st.slider("Chlorides", 0.01, 0.6, 0.05)
            free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1.0, 72.0, 30.0)
            total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6.0, 289.0, 100.0)
            density = st.slider("Density", 0.98, 1.04, 0.99, step=0.001, format="%.4f")

        with col3:
            pH = st.slider("pH", 2.7, 4.0, 3.2)
            sulphates = st.slider("Sulphates", 0.2, 2.0, 0.5)
            alcohol = st.slider("Alcohol", 8.0, 15.0, 10.0)
            wine_type = st.radio("Wine Type", ["Red", "White"], horizontal=True)
            wine_type_val = 1 if wine_type == "Red" else 0
        
        if st.button("Predict", type="primary"):
            # Prepare input
            input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                                    pH, sulphates, alcohol, wine_type_val]])
            
            try:
                input_scaled = scaler.transform(input_data)
            except Exception as e:
                st.error(f"Error transforming data: {str(e)}")
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
                        'Prediction': 'Good Quality' if pred == 1 else 'Poor Quality',
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
            good_quality_count = sum([1 for p in predictions.values() if p['Prediction'] == 'Good Quality'])
            
            st.markdown("---")
            st.subheader("Consensus Prediction")
            
            if good_quality_count >= 4:
                st.success(f"✅ **MAJORITY PREDICTION: Good Quality Wine** ({good_quality_count}/6 models)")
            elif good_quality_count >= 3:
                st.warning(f"⚠️ **SPLIT PREDICTION: Inconclusive** ({good_quality_count}/6 models predict Good Quality)")
            else:
                st.error(f"❌ **MAJORITY PREDICTION: Poor Quality Wine** ({6-good_quality_count}/6 models)")

    elif page == "Custom Data Prediction":
        st.header("📂 Custom Data Prediction")
        st.info("Upload a CSV file containing test data to evaluate models or make batch predictions.")

        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df_upload = pd.read_csv(uploaded_file, sep=';' if uploaded_file.name.endswith('.csv') else ',')
                
                # Check for target column
                target_col = 'target' if 'target' in df_upload.columns else ('quality' if 'quality' in df_upload.columns else None)
                
                # If target is quality (score), convert to binary
                if target_col == 'quality':
                     df_upload['target'] = (df_upload['quality'] >= 6).astype(int)
                     df_upload = df_upload.drop('quality', axis=1)
                     target_col = 'target'
                
                st.write("Preview of uploaded data:")
                st.dataframe(df_upload.head())
                
                # Model selection
                selected_model_upload = st.selectbox("Select Model for prediction", list(models.keys()))
                model_upload = models[selected_model_upload]
                
                # Prepare features
                expected_features = test_data['feature_names']
                
                # Check if all features exist
                missing_cols = [col for col in expected_features if col not in df_upload.columns]
                
                if missing_cols:
                    st.error(f"Missing columns in uploaded file: {missing_cols}")
                else:
                    X_upload = df_upload[expected_features]
                    
                    # Transform if needed
                    if 'Logistic' in selected_model_upload or 'Nearest' in selected_model_upload:
                         X_input = scaler.transform(X_upload)
                    else:
                         X_input = X_upload
                    
                    if st.button("Run Prediction"):
                        y_pred_upload = model_upload.predict(X_input)
                        
                        # Add predictions to dataframe
                        df_results = df_upload.copy()
                        df_results['Prediction'] = ['Good Quality' if p == 1 else 'Poor Quality' for p in y_pred_upload]
                        
                        st.subheader("Prediction Results")
                        st.dataframe(df_results)
                        
                        # If target column exists, show evaluation
                        if target_col:
                            y_true_upload = df_upload[target_col]
                            
                            st.markdown("---")
                            st.subheader("Evaluation Metrics on Uploaded Data")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Accuracy", f"{mts.accuracy_score(y_true_upload, y_pred_upload):.4f}")
                            col2.metric("Precision", f"{mts.precision_score(y_true_upload, y_pred_upload, zero_division=0):.4f}")
                            col3.metric("Recall", f"{mts.recall_score(y_true_upload, y_pred_upload, zero_division=0):.4f}")
                            col4.metric("F1 Score", f"{mts.f1_score(y_true_upload, y_pred_upload, zero_division=0):.4f}")
                            
                            col_cm, col_cr = st.columns(2)
                            
                            with col_cm:
                                st.markdown("**Confusion Matrix**")
                                st.pyplot(plot_confusion_matrix(y_true_upload, y_pred_upload))
                                
                            with col_cr:
                                st.markdown("**Classification Report**")
                                report_upload = mts.classification_report(y_true_upload, y_pred_upload, output_dict=True)
                                st.dataframe(pd.DataFrame(report_upload).transpose().style.format("{:.3f}"))
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>Machine Learning Classification Assignment | Wine Quality Prediction</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
