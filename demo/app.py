"""Streamlit demo application for medical time series analysis."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.device import get_device, set_seed
from src.models.heart_rate_models import create_model
from src.data.synthetic import SyntheticHeartRateGenerator
from src.data.preprocessing import TimeSeriesPreprocessor
from omegaconf import DictConfig, OmegaConf


# Page configuration
st.set_page_config(
    page_title="Medical Time Series Analysis",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for disclaimer
st.markdown("""
<style>
.disclaimer {
    background-color: #ffebee;
    border: 2px solid #f44336;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
    font-weight: bold;
    color: #d32f2f;
}
</style>
""", unsafe_allow_html=True)

# Disclaimer banner
st.markdown("""
<div class="disclaimer">
⚠️ DISCLAIMER: This is a research demonstration for educational purposes only. 
NOT FOR CLINICAL USE. NOT MEDICAL ADVICE. Always consult healthcare professionals for medical decisions.
</div>
""", unsafe_allow_html=True)

# Title
st.title("❤️ Medical Time Series Analysis Demo")
st.markdown("**Heart Rate Prediction using Deep Learning**")

# Sidebar
st.sidebar.header("Configuration")

# Model selection
model_architecture = st.sidebar.selectbox(
    "Model Architecture",
    ["lstm", "gru", "transformer", "cnn1d"],
    index=0
)

# Data parameters
st.sidebar.subheader("Data Parameters")
sequence_length = st.sidebar.slider("Sequence Length", 10, 100, 50)
prediction_horizon = st.sidebar.slider("Prediction Horizon", 1, 5, 1)
n_patients = st.sidebar.slider("Number of Patients", 10, 100, 20)

# Model parameters
st.sidebar.subheader("Model Parameters")
hidden_size = st.sidebar.slider("Hidden Size", 16, 128, 64)
num_layers = st.sidebar.slider("Number of Layers", 1, 4, 2)
dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)

# Training parameters
st.sidebar.subheader("Training Parameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001)
batch_size = st.sidebar.slider("Batch Size", 8, 64, 32)
num_epochs = st.sidebar.slider("Number of Epochs", 10, 100, 50)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Data Generation", "Model Training", "Predictions", "Analysis"])

with tab1:
    st.header("📊 Synthetic Data Generation")
    
    if st.button("Generate New Data"):
        # Set random seed for reproducibility
        set_seed(42)
        
        # Create configuration
        config = DictConfig({
            'data': {
                'synthetic': {
                    'n_samples': 1000,
                    'sequence_length': sequence_length,
                    'prediction_horizon': prediction_horizon,
                    'noise_level': 0.1,
                    'trend_strength': 0.5,
                    'seasonality_periods': [24, 168]
                },
                'preprocessing': {
                    'normalize': True,
                    'standardization_method': 'zscore',
                    'handle_missing': 'interpolate'
                },
                'splits': {
                    'train_ratio': 0.7,
                    'val_ratio': 0.15,
                    'test_ratio': 0.15,
                    'random_seed': 42
                }
            }
        })
        
        # Generate data
        generator = SyntheticHeartRateGenerator(config)
        df = generator.generate_dataset()
        
        # Store in session state
        st.session_state['data'] = df
        st.session_state['config'] = config
        
        st.success(f"Generated data for {df['patient_id'].nunique()} patients!")
    
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", df['patient_id'].nunique())
        with col2:
            st.metric("Total Samples", len(df))
        with col3:
            st.metric("Avg Heart Rate", f"{df['heart_rate'].mean():.1f} BPM")
        with col4:
            st.metric("Heart Rate Std", f"{df['heart_rate'].std():.1f} BPM")
        
        # Visualizations
        st.subheader("Data Visualization")
        
        # Patient selection
        selected_patient = st.selectbox(
            "Select Patient to Visualize",
            df['patient_id'].unique()[:n_patients]
        )
        
        patient_data = df[df['patient_id'] == selected_patient].sort_values('timestamp')
        
        # Heart rate time series
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=patient_data['timestamp'],
            y=patient_data['heart_rate'],
            mode='lines',
            name='Heart Rate',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f"Heart Rate Time Series - Patient {selected_patient}",
            xaxis_title="Time",
            yaxis_title="Heart Rate (BPM)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                df, 
                x='heart_rate', 
                nbins=30,
                title='Heart Rate Distribution',
                labels={'heart_rate': 'Heart Rate (BPM)', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_box = px.box(
                df, 
                y='heart_rate',
                title='Heart Rate Distribution by Patient',
                labels={'heart_rate': 'Heart Rate (BPM)'}
            )
            st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    st.header("🤖 Model Training")
    
    if 'data' not in st.session_state:
        st.warning("Please generate data first!")
    else:
        if st.button("Start Training"):
            with st.spinner("Training model..."):
                # Update configuration with sidebar values
                config = st.session_state['config']
                config.model = DictConfig({
                    'architecture': model_architecture,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'bidirectional': False,
                    'transformer': {
                        'd_model': hidden_size,
                        'nhead': 4,
                        'num_layers': num_layers,
                        'dim_feedforward': hidden_size * 2
                    },
                    'cnn1d': {
                        'channels': [32, 64, 128],
                        'kernel_sizes': [3, 5, 7],
                        'pooling': 'max'
                    }
                })
                
                config.training = DictConfig({
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs,
                    'optimizer': 'adam',
                    'scheduler': 'cosine',
                    'weight_decay': 1e-5,
                    'early_stopping': {
                        'patience': 10,
                        'min_delta': 0.001
                    },
                    'loss': {
                        'type': 'mse',
                        'huber_delta': 1.0,
                        'quantile_alpha': 0.5
                    }
                })
                
                # Preprocess data
                preprocessor = TimeSeriesPreprocessor(config)
                processed_data = preprocessor.preprocess_pipeline(st.session_state['data'])
                
                # Create model
                device = get_device()
                model = create_model(config)
                model = model.to(device)
                
                # Simple training loop (simplified for demo)
                from src.data.dataset import TimeSeriesDataModule
                data_module = TimeSeriesDataModule(processed_data, batch_size=batch_size)
                train_loader, val_loader, test_loader = data_module.get_data_loaders()
                
                # Training metrics storage
                train_losses = []
                val_losses = []
                
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                criterion = torch.nn.MSELoss()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for epoch in range(num_epochs):
                    # Training
                    model.train()
                    train_loss = 0.0
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    # Validation
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for data, target in val_loader:
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            loss = criterion(output, target)
                            val_loss += loss.item()
                    
                    train_loss /= len(train_loader)
                    val_loss /= len(val_loader)
                    
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    
                    # Update progress
                    progress = (epoch + 1) / num_epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Store trained model
                st.session_state['model'] = model
                st.session_state['train_losses'] = train_losses
                st.session_state['val_losses'] = val_losses
                st.session_state['test_loader'] = test_loader
                
                st.success("Training completed!")
                
                # Plot training curves
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=train_losses,
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    y=val_losses,
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title='Training Progress',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🔮 Predictions")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first!")
    else:
        model = st.session_state['model']
        test_loader = st.session_state['test_loader']
        
        # Get a batch for prediction
        data, target = next(iter(test_loader))
        device = get_device()
        data, target = data.to(device), target.to(device)
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            predictions = model(data)
        
        # Convert to numpy
        pred_np = predictions.cpu().numpy()
        target_np = target.cpu().numpy()
        
        # Calculate metrics
        mse = np.mean((pred_np - target_np) ** 2)
        mae = np.mean(np.abs(pred_np - target_np))
        r2 = 1 - (np.sum((target_np - pred_np) ** 2) / np.sum((target_np - np.mean(target_np)) ** 2))
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MSE", f"{mse:.4f}")
        with col2:
            st.metric("MAE", f"{mae:.4f}")
        with col3:
            st.metric("R²", f"{r2:.4f}")
        
        # Prediction vs actual plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=target_np,
            y=pred_np,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(target_np.min(), pred_np.min())
        max_val = max(target_np.max(), pred_np.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Predictions vs Actual',
            xaxis_title='Actual Heart Rate (BPM)',
            yaxis_title='Predicted Heart Rate (BPM)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample predictions
        st.subheader("Sample Predictions")
        sample_idx = st.slider("Select Sample", 0, len(pred_np)-1, 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Actual", f"{target_np[sample_idx]:.1f} BPM")
        with col2:
            st.metric("Predicted", f"{pred_np[sample_idx]:.1f} BPM")
        
        # Show input sequence
        input_seq = data[sample_idx, :, 0].cpu().numpy()
        fig_seq = go.Figure()
        fig_seq.add_trace(go.Scatter(
            y=input_seq,
            mode='lines+markers',
            name='Input Sequence',
            line=dict(color='green', width=2)
        ))
        
        fig_seq.update_layout(
            title=f'Input Sequence for Sample {sample_idx}',
            xaxis_title='Time Steps',
            yaxis_title='Heart Rate (BPM)',
            height=300
        )
        
        st.plotly_chart(fig_seq, use_container_width=True)

with tab4:
    st.header("📈 Analysis")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first!")
    else:
        st.subheader("Model Architecture")
        st.code(f"""
Model: {model_architecture.upper()}
Hidden Size: {hidden_size}
Number of Layers: {num_layers}
Dropout: {dropout}
Sequence Length: {sequence_length}
Prediction Horizon: {prediction_horizon}
        """)
        
        st.subheader("Training Configuration")
        st.code(f"""
Learning Rate: {learning_rate}
Batch Size: {batch_size}
Number of Epochs: {num_epochs}
Optimizer: Adam
Loss Function: MSE
        """)
        
        # Error analysis
        if 'train_losses' in st.session_state:
            st.subheader("Training Analysis")
            
            # Final training metrics
            final_train_loss = st.session_state['train_losses'][-1]
            final_val_loss = st.session_state['val_losses'][-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Training Loss", f"{final_train_loss:.4f}")
            with col2:
                st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
            
            # Overfitting analysis
            train_val_diff = abs(final_train_loss - final_val_loss)
            if train_val_diff < 0.01:
                st.success("✅ Good generalization - low train/val loss difference")
            elif train_val_diff < 0.05:
                st.warning("⚠️ Moderate overfitting - consider regularization")
            else:
                st.error("❌ Significant overfitting - increase regularization")
        
        st.subheader("Clinical Interpretation")
        st.markdown("""
        **Heart Rate Prediction Analysis:**
        
        - **Normal Range**: 60-100 BPM for adults at rest
        - **Clinical Significance**: Heart rate monitoring is crucial for:
          - Detecting arrhythmias
          - Monitoring medication effects
          - Assessing cardiovascular health
          - Early warning systems in ICU
        
        **Model Performance Considerations:**
        - **MSE < 25**: Good for clinical monitoring
        - **MAE < 5 BPM**: Acceptable for trend analysis
        - **R² > 0.8**: Strong predictive performance
        
        **Limitations:**
        - Synthetic data may not capture all physiological variations
        - Individual differences in heart rate patterns
        - External factors (medication, stress, exercise) not modeled
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p>Medical Time Series Analysis Demo | Research & Educational Use Only</p>
<p>⚠️ NOT FOR CLINICAL USE | Always consult healthcare professionals</p>
</div>
""", unsafe_allow_html=True)
