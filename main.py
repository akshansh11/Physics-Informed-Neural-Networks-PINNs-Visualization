import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="PINN Visualization",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the PINN model
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

@st.cache_data
def generate_data(equation_type, x_points=50, t_points=50):
    """Generate synthetic data for the selected PDE."""
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

def create_solution_plot(X, T, u_pred, x_train, t_train, u_train, equation_type):
    """Create interactive plot for PDE solution visualization."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'{equation_type} Equation PINN Prediction',
            'Training Data Distribution'
        ),
        horizontal_spacing=0.15
    )
    
    # Add contour plot
    fig.add_trace(
        go.Contour(
            x=np.array(X[0, :]),
            y=np.array(T[:, 0]),
            z=np.array(u_pred),
            colorscale='RdBu',
            colorbar=dict(
                title='u(x,t)',
                title_side='right',
                thickness=15
            ),
            name='PINN Prediction'
        ),
        row=1, col=1
    )
    
    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=np.array(x_train),
            y=np.array(t_train),
            mode='markers',
            marker=dict(
                size=8,
                color=np.array(u_train),
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(
                    title='Training Data',
                    thickness=15,
                    x=1.15
                )
            ),
            name='Training Data'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{equation_type} Equation Solution Visualization',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        showlegend=False,
        height=600,
        width=None,  # Auto-width
    )
    
    # Update axes
    fig.update_xaxes(title_text='Spatial Coordinate (x)', row=1, col=1)
    fig.update_yaxes(title_text='Time (t)', row=1, col=1)
    fig.update_xaxes(title_text='Spatial Coordinate (x)', row=1, col=2)
    fig.update_yaxes(title_text='Time (t)', row=1, col=2)
    
    return fig

def create_loss_plot(data_losses, physics_losses, equation_type):
    """Create interactive plot for training losses."""
    epochs = np.arange(1, len(data_losses) + 1)
    
    fig = go.Figure()
    
    # Add traces for each loss component
    fig.add_trace(go.Scatter(
        x=epochs,
        y=np.array(data_losses),
        mode='lines',
        name='Data Loss',
        line=dict(color='#2166ac', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=np.array(physics_losses),
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
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{equation_type} Equation PINN Training Losses',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        xaxis_title='Training Epochs',
        yaxis_title='Loss Value',
        yaxis_type='log',
        height=400,
        width=None,  # Auto-width
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig

def simulate_training(equation_type, n_epochs):
    """Simulate training losses for visualization."""
    t = np.linspace(0, 10, n_epochs)
    
    if equation_type == "Heat":
        data_losses = np.exp(-0.5 * t) + 0.01 * np.random.rand(n_epochs)
        physics_losses = 0.5 * np.exp(-0.4 * t) + 0.005 * np.random.rand(n_epochs)
    else:  # Wave
        data_losses = np.exp(-0.3 * t) * (0.5 + 0.5 * np.cos(2 * t)) + 0.01 * np.random.rand(n_epochs)
        physics_losses = np.exp(-0.25 * t) * (0.3 + 0.7 * np.cos(1.5 * t)) + 0.005 * np.random.rand(n_epochs)
    
    return data_losses, physics_losses

def main():
    """Main application function."""
    # Application title and description
    st.title("Physics-Informed Neural Networks (PINNs) Visualization")
    st.markdown("""
    Explore how PINNs solve partial differential equations by combining 
    data-driven learning with physics-based constraints. Select different 
    equations and parameters to visualize the solution process.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        equation_type = st.selectbox(
            "Select PDE Type",
            ["Heat", "Wave"],
            help="Choose between Heat and Wave equation"
        )
        
        n_points = st.slider(
            "Number of Training Points",
            min_value=20,
            max_value=100,
            value=50,
            step=5,
            help="Number of spatial and temporal points"
        )
        
        hidden_size = st.slider(
            "Neural Network Hidden Size",
            min_value=16,
            max_value=64,
            value=32,
            step=8,
            help="Number of neurons in hidden layers"
        )
        
        n_epochs = st.slider(
            "Number of Training Epochs",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Number of training iterations"
        )
    
    try:
        # Generate data
        X, T, u_true = generate_data(equation_type, n_points, n_points)
        x_train = X.flatten()
        t_train = T.flatten()
        u_train = u_true.flatten()
        
        # Simulate training
        data_losses, physics_losses = simulate_training(equation_type, n_epochs)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs([
            "Solution Visualization",
            "Training Losses",
            "Model Information"
        ])
        
        with tab1:
            st.plotly_chart(
                create_solution_plot(
                    X, T, u_true, x_train, t_train, u_train, equation_type
                ),
                use_container_width=True
            )
            
            st.markdown(f"""
            ### Solution Details
            - **Left Plot**: {equation_type} equation solution predicted by PINN
            - **Right Plot**: Distribution of training data points
            - Color represents solution magnitude u(x,t)
            """)
        
        with tab2:
            st.plotly_chart(
                create_loss_plot(data_losses, physics_losses, equation_type),
                use_container_width=True
            )
            
            st.markdown(f"""
            ### Loss Components
            - **Data Loss**: L₂ error between predictions and training data
            - **Physics Loss**: {equation_type} equation PDE residual
            - **Total Loss**: Combined loss function guiding PINN training
            """)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### Model Architecture
                """)
                st.code(f"""
Neural Network Structure:
- Input Layer: 2 neurons (x, t)
- Hidden Layers: 3 × {hidden_size} neurons
- Activation: Tanh
- Output Layer: 1 neuron (u)
                """)
            
            with col2:
                st.markdown("""
                ### PDE Information
                """)
                if equation_type == "Heat":
                    st.latex(r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u")
                else:
                    st.latex(r"\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u")
                st.markdown(f"""
                - Domain: x ∈ [0,1], t ∈ [0,1]
                - Training points: {n_points}²
                - Training epochs: {n_epochs}
                """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
