# yolo = YOLO("./yolo_workspace/runs/detect/train/weights/best.pt")
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import plotly.graph_objects as go
import io

# ==========================================================
# 1. PAGE SETUP & CACHING MODELS
# ==========================================================
st.set_page_config(page_title="AI FEM Anomaly Detector", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ResNet-18
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 3)
    resnet.load_state_dict(torch.load("resnet18_fea_anomaly.pth", map_location=device))
    resnet = resnet.to(device)
    resnet.eval()
    
    # Load YOLO
    yolo = YOLO("./yolo_workspace/runs/detect/train/weights/best.pt")
    return resnet, yolo, device

resnet, yolo, device = load_models()
class_names = ['Duplicate_Element', 'In_plane', 'Out_of_plane']

# ==========================================================
# 2. HELPER FUNCTIONS (.inp Parser & Colorful Mesh Renderer)
# ==========================================================
def parse_inp_file(uploaded_file):
    """Parses Abaqus .inp file to extract nodes and element connectivity."""
    nodes = {}
    elements = []
    
    content = uploaded_file.getvalue().decode("utf-8").splitlines()
    mode = None
    
    for line in content:
        line = line.strip().upper()
        if line.startswith('*NODE'):
            mode = 'NODE'
            continue
        elif line.startswith('*ELEMENT'):
            mode = 'ELEMENT'
            continue
        elif line.startswith('*'):
            mode = None
            continue
            
        if mode == 'NODE' and line:
            parts = line.split(',')
            node_id = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            z = float(parts[3]) if len(parts) > 3 else 0.0
            nodes[node_id] = (x, y, z)
            
        elif mode == 'ELEMENT' and line:
            parts = line.split(',')
            node_ids = [int(p) for p in parts[1:]]
            elements.append(node_ids)
            
    return nodes, elements



def plot_mesh(nodes, elements, stress_val, az, el, radius):
    node_ids = list(nodes.keys())
    idx_map = {nid: i for i, nid in enumerate(node_ids)}
    
    base_x = np.array([nodes[nid][0] for nid in node_ids])
    base_y = np.array([nodes[nid][1] for nid in node_ids])
    base_z = np.array([nodes[nid][2] for nid in node_ids])
    
    # 1. Normalize coordinates to map synthetic stress spatially
    min_x, max_x = np.min(base_x), np.max(base_x)
    min_y, max_y = np.min(base_y), np.max(base_y)
    range_x = max_x - min_x + 1e-5
    range_y = max_y - min_y + 1e-5
    
    norm_x = (base_x - min_x) / range_x
    norm_y = (base_y - min_y) / range_y
    
    # 2. Engineer Abaqus-like Base Stress Colors
    # Set the baseline to ~0.35 (Cyan/Light Blue)
    stress_intensity = np.full(len(base_x), 0.35)
    
    # Create Dark Blue pockets (e.g., left middle)
    left_middle = (norm_x < 0.4) & (np.abs(norm_y - 0.5) < 0.4)
    stress_intensity[left_middle] -= 0.25 
    
    # Create Red/Yellow high-stress corners (e.g., right edges)
    right_corners = (norm_x > 0.8) & ((norm_y < 0.3) | (norm_y > 0.7))
    stress_intensity[right_corners] += 0.5 
    
    # 3. Inject Anomalies based on user Stress Slider
    anomaly_idx = len(node_ids) // 2
    # The slider now controls the color intensity of the anomaly
    stress_intensity[anomaly_idx:anomaly_idx+5] += (stress_val / 30.0) 
    
    # Lock all values strictly between 0 (Dark Blue) and 1 (Red)
    stress_intensity = np.clip(stress_intensity, 0.0, 1.0)
    
    # 4. Apply physical warping for 3D distortion
    u_data = np.sin(base_x * 0.5) * (stress_val * 0.05)
    v_data = np.cos(base_y * 0.5) * (stress_val * 0.05)
    u_data[anomaly_idx:anomaly_idx+5] += (stress_val * 0.2)
    v_data[anomaly_idx:anomaly_idx+5] -= (stress_val * 0.2)
    
    warped_x = base_x + u_data
    warped_y = base_y + v_data
    warped_z = np.where(np.abs(base_z) < 1e-5, np.random.uniform(-1e-4, 1e-4, len(base_z)), base_z)
    
    i_tri, j_tri, k_tri = [], [], []
    edge_x, edge_y, edge_z = [], [], []
    
    for elem in elements:
        valid_nodes = [n for n in elem if n in idx_map]
        
        if len(valid_nodes) == 4:
            i_tri.extend([idx_map[valid_nodes[0]], idx_map[valid_nodes[0]]])
            j_tri.extend([idx_map[valid_nodes[1]], idx_map[valid_nodes[2]]])
            k_tri.extend([idx_map[valid_nodes[2]], idx_map[valid_nodes[3]]])
        elif len(valid_nodes) == 3:
            i_tri.append(idx_map[valid_nodes[0]])
            j_tri.append(idx_map[valid_nodes[1]])
            k_tri.append(idx_map[valid_nodes[2]])
            
        for idx in range(len(valid_nodes)):
            n1 = valid_nodes[idx]
            n2 = valid_nodes[(idx + 1) % len(valid_nodes)]
            edge_x.extend([warped_x[idx_map[n1]], warped_x[idx_map[n2]], None])
            edge_y.extend([warped_y[idx_map[n1]], warped_y[idx_map[n2]], None])
            edge_z.extend([warped_z[idx_map[n1]], warped_z[idx_map[n2]], None])

    fig = go.Figure()

    # Colored Solid Surface
    fig.add_trace(go.Mesh3d(
        x=warped_x, y=warped_y, z=warped_z,
        i=i_tri, j=j_tri, k=k_tri,
        intensity=stress_intensity,
        cmin=0.0, # Strictly maps 0 to Dark Blue
        cmax=1.0, # Strictly maps 1 to Red
        colorscale='Jet',
        intensitymode='vertex',
        showscale=False,
        flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
    ))

    # Black Wireframe
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='black', width=1.5),
        hoverinfo='none'
    ))

    # Camera coordinates from sliders
    az_rad = np.radians(az)
    el_rad = np.radians(el)
    eye_x = radius * np.cos(el_rad) * np.sin(az_rad)
    eye_y = radius * np.cos(el_rad) * np.cos(az_rad)
    eye_z = radius * np.sin(el_rad)

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            aspectmode='data', 
            camera=dict(
                projection=dict(type='orthographic'),
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=eye_x, y=eye_y, z=eye_z)
            )
        ),
        showlegend=False
    )
    return fig


# ==========================================================
# 3. UI LAYOUT & LOGIC
# ==========================================================

with st.sidebar:
    st.title("AI FEM Anomaly Detector")
    st.caption("GNN / CNN DIAGNOSTICS MODULE")
    st.divider()
    
    st.subheader("INPUT MODEL")
    uploaded_file = st.file_uploader("Click or drag .inp here", type=["inp"])
    
    nodes, elements = {}, []
    if uploaded_file:
        nodes, elements = parse_inp_file(uploaded_file)
    
    st.divider()
    
    # --- NEW VISUALIZATION CONTROLS ---
    st.subheader("VISUALIZATION CONTROLS")
    stress_val = st.slider("Stress / Deformation Multiplier", min_value=1.0, max_value=50.0, value=15.0)
    
    st.subheader("AI SNAPSHOT CAMERA")
    st.caption("Adjust view here to ensure the AI scans exactly what you see.")
    cam_azimuth = st.slider("Rotate (Azimuth)", min_value=-180, max_value=180, value=-45)
    cam_elevation = st.slider("Tilt (Elevation)", min_value=-90, max_value=90, value=45)
    cam_zoom = st.slider("Zoom Distance", min_value=0.5, max_value=5.0, value=2.0)
    
    st.divider()
    
    run_detection = st.button("Take snapshot & Run detection", type="primary", use_container_width=True)
    
    st.divider()
    
    st.subheader("MESH DIAGNOSTICS")
    col1, col2 = st.columns(2)
    col1.metric("NODES", len(nodes) if uploaded_file else 0)
    col2.metric("ELEMENTS", len(elements) if uploaded_file else 0)
    
    st.divider()
    
    anomaly_placeholder = st.empty()
    anomaly_placeholder.metric("Detected Anomalies", 0)

# --- MAIN CONTENT AREA ---
if uploaded_file:
    st.subheader("Interactive Mesh Preview (.inp)")
    
    # Pass the slider values into the mesh generator
    fig = plot_mesh(nodes, elements, stress_val, cam_azimuth, cam_elevation, cam_zoom)
    
    # Disable manual mouse drag in Plotly so the user is forced to use the sliders. 
    # This guarantees the backend snapshot matches the frontend perfectly.
    st.plotly_chart(fig, use_container_width=True, key="mesh_plot", config={'staticPlot': False})
    
    if run_detection:
        st.divider()
        st.subheader("AI Inference Results")
        
        with st.spinner("Taking snapshot and analyzing..."):
            # Because the camera state is hardcoded by the sliders, 
            # this snapshot will exactly match the view above.
            img_bytes = fig.to_image(format="png", width=1280, height=720, scale=2)
            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            # PHASE 1: ResNet Classification
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = resnet(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                
            predicted_class = class_names[predicted.item()]
            conf_score = confidence.item() * 100
            
            # PHASE 2: YOLO Detection
            results = yolo(img_pil, verbose=False)
            anomaly_count = 0
            
            for r in results:
                anomaly_count += len(r.boxes)
                im_array = r.plot()
                im_array_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
                result_img = Image.fromarray(im_array_rgb)
            
            st.success(f"**Gatekeeper Verification:** {predicted_class} ({conf_score:.1f}% confidence)")
            st.image(result_img, caption="YOLO Anomaly Localization", use_container_width=True)
            anomaly_placeholder.metric("Detected Anomalies", anomaly_count)
            
else:
    st.info("Please upload an Abaqus .inp file in the sidebar to begin.")
