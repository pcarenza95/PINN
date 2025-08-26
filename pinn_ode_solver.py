import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Callable, Dict, List, Tuple
import time
import sympy as sp
import io

# Set page config
st.set_page_config(
    page_title="PINN ODE Solver",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class PINN_ODE_Solver:
    def __init__(self, ode_func: Callable, ic_conditions: Dict[str, float], 
                 domain: Tuple[float, float] = (0, 1), 
                 hidden_layers: List[int] = [64, 64, 64],
                 activation: str = 'tanh'):
        self.ode_func = ode_func
        self.ic_conditions = ic_conditions
        self.domain = domain
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.order = self._determine_order()
        self.model = self._build_model()
        self.x_ic = tf.constant([[domain[0]]], dtype=tf.float32)
        
    def _determine_order(self) -> int:
        order = 0
        if 'y' in self.ic_conditions:
            order = max(order, 1)
        if 'dy_dx' in self.ic_conditions:
            order = max(order, 2)
        if 'd2y_dx2' in self.ic_conditions:
            order = max(order, 3)
        return order
    
    def _build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(1,)))
        
        for units in self.hidden_layers:
            model.add(tf.keras.layers.Dense(
                units, 
                activation=self.activation,
                kernel_initializer='glorot_normal',
                bias_initializer='zeros'
            ))
        
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        return model
    
    def _compute_derivatives(self, x: tf.Tensor, order: int = 2) -> List[tf.Tensor]:
        derivatives = []
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y = self.model(x)
            derivatives.append(y)
            
            current = y
            for i in range(1, order + 1):
                current = tape.gradient(current, x)
                if current is None:
                    current = tf.zeros_like(x)
                derivatives.append(current)
        
        return derivatives
    
    def physics_loss(self, x_collocation: tf.Tensor) -> tf.Tensor:
        derivatives = self._compute_derivatives(x_collocation, order=self.order)
        y = derivatives[0]
        dy_dx = derivatives[1] if len(derivatives) > 1 else None
        d2y_dx2 = derivatives[2] if len(derivatives) > 2 else None
        
        residual = self.ode_func(x_collocation, y, dy_dx, d2y_dx2)
        return tf.reduce_mean(tf.square(residual))
    
    def initial_condition_loss(self) -> tf.Tensor:
        ic_loss = 0.0
        derivatives = self._compute_derivatives(self.x_ic, order=self.order)
        
        if 'y' in self.ic_conditions:
            y_pred = derivatives[0]
            ic_loss += tf.reduce_mean(tf.square(y_pred - self.ic_conditions['y']))
        
        if 'dy_dx' in self.ic_conditions and len(derivatives) > 1:
            dy_dx_pred = derivatives[1]
            ic_loss += tf.reduce_mean(tf.square(dy_dx_pred - self.ic_conditions['dy_dx']))
        
        if 'd2y_dx2' in self.ic_conditions and len(derivatives) > 2:
            d2y_dx2_pred = derivatives[2]
            ic_loss += tf.reduce_mean(tf.square(d2y_dx2_pred - self.ic_conditions['d2y_dx2']))
        
        return ic_loss
    
    def total_loss(self, x_collocation: tf.Tensor) -> tf.Tensor:
        physics = self.physics_loss(x_collocation)
        ic = self.initial_condition_loss()
        return physics + 10.0 * ic
    
    @tf.function
    def train_step(self, x_collocation: tf.Tensor, optimizer: tf.keras.optimizers.Optimizer) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self.total_loss(x_collocation)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    def train(self, n_collocation: int = 1000, n_epochs: int = 10000, 
              learning_rate: float = 0.001, progress_callback=None) -> List[float]:
        x_uniform = np.linspace(self.domain[0], self.domain[1], n_collocation // 2)
        x_boundary = np.random.uniform(self.domain[0], self.domain[1], n_collocation // 4)
        x_ic_region = np.random.uniform(self.domain[0], self.domain[0] + 0.1*(self.domain[1]-self.domain[0]), n_collocation // 4)
        
        x_collocation = np.concatenate([x_uniform, x_boundary, x_ic_region])
        x_collocation = np.sort(x_collocation).reshape(-1, 1)
        x_collocation_tf = tf.convert_to_tensor(x_collocation, dtype=tf.float32)
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=2000,
            decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        losses = []
        best_loss = float('inf')
        patience = 1000
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            loss = self.train_step(x_collocation_tf, optimizer)
            loss_value = loss.numpy()
            losses.append(loss_value)
            
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience and epoch > 1000:
                break
            
            if progress_callback and epoch % 100 == 0:
                progress_callback(epoch, n_epochs, loss_value)
        
        training_time = time.time() - start_time
        return losses, training_time
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        x_tensor = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float32)
        return self.model(x_tensor).numpy().flatten()
    
    def solve_ode_numerical(self, ode_system: Callable, x_eval: np.ndarray) -> np.ndarray:
        y0 = []
        if 'y' in self.ic_conditions:
            y0.append(self.ic_conditions['y'])
        if 'dy_dx' in self.ic_conditions and self.order >= 2:
            y0.append(self.ic_conditions['dy_dx'])
        
        sol = solve_ivp(ode_system, self.domain, y0, t_eval=x_eval, method='RK45', rtol=1e-8, atol=1e-8)
        return sol.y[0]

# Predefined ODE examples
ODE_EXAMPLES = {
    "Linear ODE: y' + y = 0": {
        "ode_func": lambda x, y, dy_dx, d2y_dx2=None: dy_dx + y,
        "ode_system": lambda x, y: -y,
        "exact_solution": lambda x: np.exp(-x),
        "default_ic": {"y": 1.0},
        "default_domain": (0, 5)
    },
    "Harmonic Oscillator: y'' + y = 0": {
        "ode_func": lambda x, y, dy_dx, d2y_dx2: d2y_dx2 + y,
        "ode_system": lambda x, y: [y[1], -y[0]],
        "exact_solution": lambda x: np.sin(x),
        "default_ic": {"y": 0.0, "dy_dx": 1.0},
        "default_domain": (0, 4*np.pi)
    },
    "Nonlinear ODE: y' + y² = 0": {
        "ode_func": lambda x, y, dy_dx, d2y_dx2=None: dy_dx + y**2,
        "ode_system": lambda x, y: -y**2,
        "exact_solution": lambda x: 1 / (1 + x),
        "default_ic": {"y": 1.0},
        "default_domain": (0, 2)
    },
    "Damped Oscillator: y'' + 0.1y' + y = 0": {
        "ode_func": lambda x, y, dy_dx, d2y_dx2: d2y_dx2 + 0.1*dy_dx + y,
        "ode_system": lambda x, y: [y[1], -0.1*y[1] - y[0]],
        "exact_solution": lambda x: np.exp(-0.05*x) * np.sin(np.sqrt(0.9975)*x),
        "default_ic": {"y": 0.0, "dy_dx": 1.0},
        "default_domain": (0, 20)
    }
}

def create_custom_ode(ode_expr, progress_bar):
    """Create custom ODE function from string expression"""
    try:
        x, y, dy_dx, d2y_dx2 = sp.symbols('x y dy_dx d2y_dx2')
        expr = sp.sympify(ode_expr)
        ode_func = sp.lambdify((x, y, dy_dx, d2y_dx2), expr, 'numpy')
        progress_bar.success("✅ ODE parsed successfully!")
        return lambda x, y, dy_dx, d2y_dx2: ode_func(x, y, dy_dx, d2y_dx2)
    except Exception as e:
        progress_bar.error(f"❌ Error parsing ODE: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🧠 Physics-Informed Neural Network ODE Solver</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<h2 class="sub-header">Configuration</h2>', unsafe_allow_html=True)
        
        # ODE selection
        ode_choice = st.selectbox(
            "Choose ODE Example:",
            list(ODE_EXAMPLES.keys()) + ["Custom ODE"]
        )
        
        if ode_choice == "Custom ODE":
            st.markdown("### Custom ODE Definition")
            st.markdown("Use symbols: `x`, `y`, `dy_dx`, `d2y_dx2`")
            custom_ode_expr = st.text_input("ODE Expression (e.g., dy_dx + y):", "dy_dx + y")
            parse_progress = st.empty()
            custom_ode_func = create_custom_ode(custom_ode_expr, parse_progress)
        else:
            custom_ode_func = None
        
        # Initial conditions
        st.markdown("### Initial Conditions")
        col1, col2, col3 = st.columns(3)
        with col1:
            y0 = st.number_input("y(0)", value=1.0, step=0.1)
        with col2:
            dy_dx0 = st.number_input("y'(0)", value=0.0, step=0.1)
        with col3:
            d2y_dx20 = st.number_input("y''(0)", value=0.0, step=0.1)
        
        ic_conditions = {}
        if y0 != 0.0:
            ic_conditions['y'] = y0
        if dy_dx0 != 0.0:
            ic_conditions['dy_dx'] = dy_dx0
        if d2y_dx20 != 0.0:
            ic_conditions['d2y_dx2'] = d2y_dx20
        
        # Domain
        st.markdown("### Domain")
        x_min = st.number_input("x_min", value=0.0, step=0.1)
        x_max = st.number_input("x_max", value=5.0, step=0.1)
        domain = (x_min, x_max)
        
        # Neural network parameters
        st.markdown("### Neural Network Parameters")
        hidden_layers = st.text_input("Hidden Layers (comma-separated)", "64,64,64")
        hidden_layers = [int(x.strip()) for x in hidden_layers.split(",") if x.strip().isdigit()]
        activation = st.selectbox("Activation Function", ["tanh", "relu", "sigmoid"])
        
        # Training parameters
        st.markdown("### Training Parameters")
        n_collocation = st.slider("Collocation Points", 100, 5000, 1000)
        n_epochs = st.slider("Epochs", 1000, 50000, 10000)
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001)
        
        # Solve button
        solve_button = st.button("🚀 Solve ODE", type="primary", use_container_width=True)
    
    # Main content area
    if solve_button:
        # Get ODE function
        if ode_choice == "Custom ODE" and custom_ode_func:
            ode_func = custom_ode_func
            ode_system = None  # Cannot provide numerical solution for custom ODE
            exact_solution = None
        elif ode_choice != "Custom ODE":
            example = ODE_EXAMPLES[ode_choice]
            ode_func = example["ode_func"]
            ode_system = example["ode_system"]
            exact_solution = example["exact_solution"]
        else:
            st.error("Please provide a valid ODE expression")
            return
        
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_placeholder = st.empty()
        
        def update_progress(epoch, total_epochs, loss):
            progress = epoch / total_epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4e}")
        
        # Create and train PINN
        with st.spinner("Training PINN..."):
            try:
                pinn = PINN_ODE_Solver(
                    ode_func=ode_func,
                    ic_conditions=ic_conditions,
                    domain=domain,
                    hidden_layers=hidden_layers,
                    activation=activation
                )
                
                losses, training_time = pinn.train(
                    n_collocation=n_collocation,
                    n_epochs=n_epochs,
                    learning_rate=learning_rate,
                    progress_callback=update_progress
                )
                
                progress_bar.progress(1.0)
                status_text.text(f"Training completed in {training_time:.2f} seconds!")
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                return
        
        # Results section
        st.markdown('<div class="success-box">✅ Training Completed Successfully!</div>', unsafe_allow_html=True)
        
        # Generate evaluation points
        x_eval = np.linspace(domain[0], domain[1], 500)
        y_pinn = pinn.predict(x_eval)
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Solutions
        ax1.plot(x_eval, y_pinn, 'r-', label='PINN Solution', linewidth=2)
        
        if exact_solution:
            y_exact = exact_solution(x_eval)
            ax1.plot(x_eval, y_exact, 'k--', label='Exact Solution', linewidth=2)
            
            if ode_system:
                y_numerical = pinn.solve_ode_numerical(ode_system, x_eval)
                ax1.plot(x_eval, y_numerical, 'b:', label='Numerical Solution', linewidth=2)
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y(x)')
        ax1.set_title('Solution Comparison')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Error
        if exact_solution:
            error = np.abs(y_pinn - y_exact)
            ax2.semilogy(x_eval, error, 'r-', label='PINN Error', linewidth=2)
            
            if ode_system:
                numerical_error = np.abs(y_numerical - y_exact)
                ax2.semilogy(x_eval, numerical_error, 'b-', label='Numerical Error', linewidth=2)
            
            ax2.set_xlabel('x')
            ax2.set_ylabel('Absolute Error')
            ax2.set_title('Error Comparison')
            ax2.legend()
            ax2.grid(True)
        
        # Plot 3: Loss history
        ax3.semilogy(losses, 'b-')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss History')
        ax3.grid(True)
        
        # Plot 4: Residual
        x_res = np.linspace(domain[0], domain[1], 100)
        x_tf = tf.convert_to_tensor(x_res.reshape(-1, 1), dtype=tf.float32)
        residuals = []
        for x_point in x_res:
            x_point_tf = tf.constant([[x_point]], dtype=tf.float32)
            residual = pinn.physics_loss(x_point_tf).numpy()
            residuals.append(residual)
        
        ax4.semilogy(x_res, residuals, 'go', markersize=2)
        ax4.set_xlabel('x')
        ax4.set_ylabel('Residual')
        ax4.set_title('ODE Residual at Collocation Points')
        ax4.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Time", f"{training_time:.2f} seconds")
        
        with col2:
            if exact_solution:
                max_error = np.max(error) if exact_solution else "N/A"
                st.metric("Max Error", f"{max_error:.2e}")
        
        with col3:
            if exact_solution:
                mean_error = np.mean(error) if exact_solution else "N/A"
                st.metric("Mean Error", f"{mean_error:.2e}")
        
        # Download results
        if exact_solution:
            results_data = np.column_stack((x_eval, y_pinn, y_exact))
            if ode_system:
                results_data = np.column_stack((results_data, y_numerical))
            
            np.savetxt("pinn_results.csv", results_data, delimiter=",", 
                      header="x,PINN,Exact,Numerical" if ode_system else "x,PINN,Exact")
        else:
            results_data = np.column_stack((x_eval, y_pinn))
            np.savetxt("pinn_results.csv", results_data, delimiter=",", header="x,PINN")
        
        with open("pinn_results.csv", "rb") as f:
            st.download_button(
                label="📥 Download Results",
                data=f,
                file_name="pinn_results.csv",
                mime="text/csv"
            )
    
    else:
        # Show instructions when not solving
        st.markdown("""
        <div class="info-box">
        <h3>📋 Instructions:</h3>
        <ol>
            <li>Select an ODE example or create a custom one</li>
            <li>Set initial conditions in the sidebar</li>
            <li>Configure the domain and neural network parameters</li>
            <li>Click "Solve ODE" to train the PINN</li>
            <li>View results and download data</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Show example ODEs
        st.markdown('<h2 class="sub-header">📚 Example ODEs</h2>', unsafe_allow_html=True)
        
        for ode_name, ode_info in ODE_EXAMPLES.items():
            with st.expander(ode_name):
                st.write(f"**Equation:** {ode_name}")
                st.write(f"**Default IC:** {ode_info['default_ic']}")
                st.write(f"**Default Domain:** {ode_info['default_domain']}")
                
                # Show sample plot
                x_sample = np.linspace(ode_info['default_domain'][0], ode_info['default_domain'][1], 100)
                y_sample = ode_info['exact_solution'](x_sample)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(x_sample, y_sample, 'b-', linewidth=2)
                ax.set_xlabel('x')
                ax.set_ylabel('y(x)')
                ax.set_title(f'Sample Solution: {ode_name}')
                ax.grid(True)
                st.pyplot(fig)

if __name__ == "__main__":
    main()