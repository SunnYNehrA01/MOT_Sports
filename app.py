import streamlit as st
import tempfile
import os
import subprocess
from core.engine import TrackingEngine

# --- Page Configuration ---
st.set_page_config(
    page_title="MOT Sports AI", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Professional Styling ---
st.markdown("""
    <style>
    /* Global Theme */
    .stApp {
        background-color: #0b0e11;
        color: #ffffff;
    }

    /* Bento Box Design */
    .bento-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Professional Emerald Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #00ff88 !important;
        color: #0b0e11 !important;
        font-weight: 700;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00cc6d !important;
    }

    /* Comparison Table Styling */
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 0.8rem;
    }
    .comparison-table th {
        text-align: left;
        color: #00ff88;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 5px;
    }
    .comparison-table td {
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    [data-testid="stSidebar"] {
        background-color: #0d1117;
    }
    
    h1, h2, h3 { color: #ffffff !important; }
    .stCaption { color: rgba(255, 255, 255, 0.5); }
    </style>
    """, unsafe_allow_html=True)

def convert_to_h264(input_path, output_path):
    """Converts OpenCV output to browser-compatible H.264."""
    command = f'ffmpeg -y -i "{input_path}" -c:v libx264 -crf 23 -preset fast -pix_fmt yuv420p "{output_path}"'
    subprocess.call(command, shell=True)

def main():
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'output_file' not in st.session_state:
        st.session_state.output_file = None

    # --- VIEW 1: Upload and Configuration ---
    if not st.session_state.processed:
        _, center_col, _ = st.columns([1, 4, 1])

        with center_col:
            st.title("MOT Sports Tracker")
            st.markdown("Advanced Identity Persistence and Tactical Analysis")
            st.divider()

            with st.sidebar:
                st.header("Pipeline Configuration")
                sport_type = st.selectbox("Sport Category", ["Basketball", "Football", "Cricket", "Racing", "General"])
                
                st.subheader("Model Selection")
                model_size = st.select_slider(
                    "Architecture", 
                    options=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"], 
                    value="yolo11s.pt"
                )

                # --- Comparison Table ---
                st.markdown("""
                <table class="comparison-table">
                    <tr><th>Model</th><th>Params</th><th>Speed</th><th>Accuracy</th></tr>
                    <tr><td>Nano (n)</td><td>2.6M</td><td>Fastest</td><td>Standard</td></tr>
                    <tr><td>Small (s)</td><td>9.4M</td><td>Balanced</td><td>High</td></tr>
                    <tr><td>Medium (m)</td><td>20.1M</td><td>Detailed</td><td>Maximum</td></tr>
                </table>
                """, unsafe_allow_html=True)

                with st.expander("Advanced Parameters"):
                    use_reid = st.checkbox("Enable Visual Re-ID", value=True)
                    use_cmc = st.checkbox("Camera Motion Compensation", value=True)
                    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.25)

            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload video file (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
            st.markdown('</div>', unsafe_allow_html=True)

            if uploaded_file:
                st.markdown('<div class="bento-card">', unsafe_allow_html=True)
                st.write("Processing Information")
                st.caption(f"File: {uploaded_file.name} | Model: {model_size}")
                
                if st.button("Run Tracking Engine"):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    input_path = tfile.name
                    tfile.close()

                    args = {
                        'model': model_size, 'use_reid': use_reid, 'use_cmc': use_cmc,
                        'conf': conf_thresh, 'high_thresh': 0.6, 'low_thresh': 0.1, 'max_lost': 60
                    }
                    
                    engine = TrackingEngine(args)
                    raw_out = f"raw_{uploaded_file.name}"
                    web_out = f"web_{uploaded_file.name}"

                    p_bar = st.progress(0)
                    try:
                        with st.spinner("Analyzing frames and resolving identities..."):
                            engine.process_video(input_path, raw_out, sport_type, lambda p: p_bar.progress(p))
                        
                        convert_to_h264(raw_out, web_out)
                        
                        if os.path.exists(raw_out): os.remove(raw_out)
                        if os.path.exists(input_path): os.remove(input_path)
                        
                        st.session_state.output_file = web_out
                        st.session_state.processed = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"Processing Error: {e}")
                st.markdown('</div>', unsafe_allow_html=True)

    # --- VIEW 2: Results View ---
    else:
        _, res_col, _ = st.columns([1, 6, 1])

        with res_col:
            st.title("Analysis Complete")
            st.caption("Temporal Consistency and ID Persistence Active")
            st.divider()

            col_vid, col_meta = st.columns([3, 2])

            with col_vid:
                st.markdown('<div class="bento-card">', unsafe_allow_html=True)
                st.video(st.session_state.output_file)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_meta:
                st.markdown('<div class="bento-card">', unsafe_allow_html=True)
                st.write("Export Metadata")
                st.info("The output video is encoded in H.264 for full browser and player support.")
                
                with open(st.session_state.output_file, "rb") as f:
                    st.download_button(
                        label="Download Analysis Video",
                        data=f,
                        file_name=st.session_state.output_file,
                        mime="video/mp4"
                    )
                st.markdown('</div>', unsafe_allow_html=True)

                if st.button("New Project"):
                    if os.path.exists(st.session_state.output_file):
                        os.remove(st.session_state.output_file)
                    st.session_state.processed = False
                    st.session_state.output_file = None
                    st.rerun()

if __name__ == "__main__":
    main()