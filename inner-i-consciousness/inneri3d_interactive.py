import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import plotly.graph_objects as go
import os

# -------------------------------
# Constants
# -------------------------------
VEDIC_STATES = [
    "Waking",
    "Sleeping",
    "Dreaming",
    "Transcendental Consciousness",
    "Cosmic Consciousness",
    "God Consciousness",
    "Unity Consciousness"
]

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------
# Step 1: Build the Model
# -------------------------------
def build_inner_i_model(input_dim):
    input_layer = Input(shape=(input_dim,), name="Precomputed_Input")

    # Foundational Inner 'I' layer
    inner_i_layer = Dense(256, activation="relu", name="Inner_I_Consciousness")(input_layer)

    # Seven layers for states of consciousness
    hidden_layer = Dense(128, activation="relu", name="Waking")(inner_i_layer)
    hidden_layer = Dropout(0.2)(hidden_layer)

    hidden_layer = Dense(96, activation="relu", name="Sleeping")(hidden_layer)
    hidden_layer = Dropout(0.2)(hidden_layer)

    hidden_layer = Dense(64, activation="relu", name="Dreaming")(hidden_layer)
    hidden_layer = Dropout(0.2)(hidden_layer)

    hidden_layer = Dense(48, activation="relu", name="Transcendental_Consciousness")(hidden_layer)
    hidden_layer = Dropout(0.2)(hidden_layer)

    hidden_layer = Dense(32, activation="relu", name="Cosmic_Consciousness")(hidden_layer)
    hidden_layer = Dropout(0.2)(hidden_layer)

    hidden_layer = Dense(24, activation="relu", name="God_Consciousness")(hidden_layer)
    hidden_layer = Dropout(0.2)(hidden_layer)

    hidden_layer = Dense(16, activation="relu", name="Unity_Consciousness")(hidden_layer)
    hidden_layer = Dropout(0.2)(hidden_layer)

    # Output layer
    output_layer = Dense(7, activation="softmax", name="Consciousness_Output")(hidden_layer)

    # Build and compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# -------------------------------
# Step 2: Visualize in 3D with Plotly
# -------------------------------
def plot_3d_inner_i_network_with_info(layer_sizes, save_path="interactive_network.html"):
    """
    Create an interactive 3D visualization of the neural network architecture
    with detailed hover info and Inner 'I' Consciousness as the foundational layer.

    Args:
    - layer_sizes: list, number of neurons in each layer (e.g., [256, 128, 96, 64, 48, 32, 24, 16]).
    - save_path: str, path to save the interactive HTML visualization.
    """
    fig = go.Figure()

    # Position and label nodes
    current_radius = 1.0
    node_positions = {"Inner I": (0, 0, 0)}
    neuron_details = []  # To store hover information for each neuron
    x_coords, y_coords, z_coords, labels = [], [], [], []

    # Add the central node
    x_coords.append(0)
    y_coords.append(0)
    z_coords.append(0)
    labels.append("Inner I Consciousness")
    neuron_details.append("Central Foundational Observer")

    for layer_idx, num_nodes in enumerate(layer_sizes):
        angle_step = 2 * np.pi / num_nodes
        z_offset = layer_idx * 2.0  # Increment Z for layering

        for node_idx in range(num_nodes):
            # Calculate radial position
            x = current_radius * np.cos(node_idx * angle_step)
            y = current_radius * np.sin(node_idx * angle_step)
            z = z_offset

            node_name = f"Layer {layer_idx + 1} - Node {node_idx + 1}"
            state = VEDIC_STATES[layer_idx] if layer_idx < len(VEDIC_STATES) else "Unknown State"

            # Store position and labels
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            labels.append(node_name)
            neuron_details.append(f"Layer: {layer_idx + 1}<br>State: {state}<br>Neuron: {node_idx + 1}")

        current_radius += 1.5  # Increment radius for next layer

    # Add connections
    edge_x, edge_y, edge_z = [], [], []
    for i in range(len(layer_sizes) - 1):
        prev_layer_start = sum(layer_sizes[:i])
        current_layer_start = sum(layer_sizes[:i + 1])
        prev_layer_end = current_layer_start
        current_layer_end = current_layer_start + layer_sizes[i + 1]

        # Fully connect layers
        for j in range(prev_layer_start, prev_layer_end):
            for k in range(current_layer_start, current_layer_end):
                edge_x += [x_coords[j], x_coords[k], None]
                edge_y += [y_coords[j], y_coords[k], None]
                edge_z += [z_coords[j], z_coords[k], None]

    # Plot nodes
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode="markers",
        marker=dict(size=5, color="blue"),
        text=neuron_details,  # Hover info
        hoverinfo="text",
        name="Neurons"
    ))

    # Plot edges
    fig.add_trace(go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="gray", width=0.5),
        hoverinfo="none",
        name="Connections"
    ))

    # Highlight the Inner 'I' node
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=10, color="gold"),
        text=["Inner I Consciousness<br>Central Foundational Observer"],
        hoverinfo="text",
        name="Inner I"
    ))

    # Layout settings
    fig.update_layout(
        title="Interactive 3D Neural Network: Inner 'I' Consciousness",
        scene=dict(
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            zaxis_title="Z-axis",
            xaxis=dict(backgroundcolor="black"),
            yaxis=dict(backgroundcolor="black"),
            zaxis=dict(backgroundcolor="black"),
        ),
        template="plotly_dark",
        showlegend=True
    )

    # Save the interactive HTML file
    fig.write_html(save_path)
    print(f"Interactive 3D visualization saved to {save_path}")

    # Display the plot in a browser
    fig.show()

# -------------------------------
# Example Usage
# -------------------------------
# Define layer sizes
layer_sizes = [256, 128, 96, 64, 48, 32, 24, 16]

# Build and visualize the model
input_dim = 10  # Example input dimensions
model = build_inner_i_model(input_dim)
plot_3d_inner_i_network_with_info(layer_sizes, save_path=os.path.join(RESULTS_DIR, "inner_i_network.html"))
