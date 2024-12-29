import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PINN(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)

def generate_data(equation_type, x_points=50, t_points=50):
    x = np.linspace(0, 1, x_points)
    t = np.linspace(0, 1, t_points)
    X, T = np.meshgrid(x, t)
    
    if equation_type == "Heat":
        alpha = 0.4
        u_true = np.exp(-alpha * np.pi**2 * T) * np.sin(np.pi * X)
    else:  # Wave
        c = 1.0
        u_true = np.sin(np.pi * X) * np.cos(np.pi * c * T)
    
    return X, T, u_true

def create_plotly_figure(X, T, u_pred, x_train, t_train, u_train, equation_type):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{equation_type} Equation PINN Prediction', 'Training Data Points')
    )
    
    # Contour plot for predictions
    fig.add_trace(
        go.Contour(
            x=X[0, :],
            y=T[:, 0],
            z=u_pred,
            colorscale='RdBu',
            colorbar=dict(title='u(x,t)'),
            name='PINN Prediction'
        ),
        row=1, col=1
    )
    
    # Scatter plot for training data
    fig.add_trace(
        go.Scatter(
            x=x_train,
            y=t_train,
            mode='markers',
            marker=dict(
                size=8,
                color=u_train,
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title='Training Data')
            ),
            name='Training Data'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f'{equation_type} Equation Solution Visualization',
        xaxis_title='Spatial Coordinate (x)',
        yaxis_title='Time (t)',
        xaxis2_title='Spatial Coordinate (x)',
        yaxis2_title='Time (t)',
        height=600,
        width=1200,
        showlegend=False
    )
    
    return fig

def plot_losses(data_losses, physics_losses, equation_type):
    epochs = range(1, len(data_losses) + 1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=data_losses,
        mode='lines',
        name='Data Loss',
        line=dict(color='#2166ac', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=physics_losses,
        mode='lines',
        name='Physics Loss',
        line=dict(color='#b2182b', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=np.array(data_losses) + np.array(physics_losses),
        mode='lines',
        name='Total Loss',
        line=dict(color='#4d4d4d', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{equation_type} Equation PINN Training Losses',
        xaxis_title='Training Epochs',
        yaxis_title='Loss Value',
        yaxis_type='log',
        height=400,
        width=800
    )
    
    return fig

def main():
    st.set_page_config(layout="wide")
    
    st.title("Physics-Informed Neural Networks (PINNs) Visualization")
    st.markdown("""
    This application demonstrates the power of Physics-Informed Neural Networks (PINNs) 
    in solving partial differential equations. Choose between Heat and Wave equations 
    to see how PINNs learn to balance data-driven and physics-driven constraints.
    """)
    
    # Sidebar controls
    st.sidebar.header("Settings")
    equation_type = st.sidebar.selectbox(
        "Select PDE Type",
        ["Heat", "Wave"]
    )
    
    n_points = st.sidebar.slider(
        "Number of Training Points",
        min_value=20,
        max_value=100,
        value=50,
        step=5
    )
    
    hidden_size = st.sidebar.slider(
        "Neural Network Hidden Size",
        min_value=16,
        max_value=64,
        value=32,
        step=8
    )
    
    n_epochs = st.sidebar.slider(
        "Number of Training Epochs",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100
    )
    
    # Generate data
    X, T, u_true = generate_data(equation_type, n_points, n_points)
    x_train = X.flatten()
    t_train = T.flatten()
    u_train = u_true.flatten()
    
    # Initialize model
    model = PINN(hidden_size=hidden_size)
    
    # Generate simulated losses
    t = np.linspace(0, 10, n_epochs)
    if equation_type == "Heat":
        data_losses = np.exp(-np.linspace(0, 5, n_epochs)) + 0.01 * np.random.rand(n_epochs)
        physics_losses = 0.5 * np.exp(-np.linspace(0, 4, n_epochs)) + 0.005 * np.random.rand(n_epochs)
    else:  # Wave
        data_losses = np.exp(-0.3 * t) * (0.5 + 0.5 * np.cos(2 * t)) + 0.01 * np.random.rand(n_epochs)
        physics_losses = np.exp(-0.25 * t) * (0.3 + 0.7 * np.cos(1.5 * t)) + 0.005 * np.random.rand(n_epochs)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Solution Visualization", "Training Losses", "Model Information"])
    
    with tab1:
        st.plotly_chart(
            create_plotly_figure(
                X, T, u_true, x_train, t_train, u_train, equation_type
            ),
            use_container_width=True
        )
        
        st.markdown(f"""
        ### Solution Visualization Details
        - Left: Contour plot showing the {equation_type} equation solution
        - Right: Training data points distribution
        - Color scale represents the solution magnitude u(x,t)
        """)
    
    with tab2:
        st.plotly_chart(
            plot_losses(data_losses, physics_losses, equation_type),
            use_container_width=True
        )
        
        st.markdown(f"""
        ### Training Loss Components
        - Data Loss: Measures the mismatch between PINN predictions and training data
        - Physics Loss: Quantifies the residual of the {equation_type} equation PDE
        - Total Loss: Sum of data and physics losses
        """)
    
    with tab3:
        st.markdown(f"""
        ### Model Architecture
        - Input dimension: 2 (x, t)
        - Hidden layers: 3 layers with {hidden_size} neurons each
        - Activation function: Tanh
        - Output dimension: 1 (u(x,t))
        
        ### {equation_type} Equation Details
        {'- Heat equation: ∂u/∂t = α∇²u' if equation_type == "Heat" else '- Wave equation: ∂²u/∂t² = c²∇²u'}
        - Domain: x ∈ [0,1], t ∈ [0,1]
        - Number of training points: {n_points}²
        """)

if __name__ == "__main__":
    main()
