import streamlit as st
import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement

# Set allowed file extensions
ALLOWED_EXTENSIONS = {'wav'}

# Load pre-trained enhancement model
enhance_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank",
)

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to process uploaded file
def process_file(file):
    # Load audio and check sampling rate
    noisy, rate = torchaudio.load(file)
    assert rate == 16000, "sampling rate must be 16000"
    # Remove extra dimension from input signal
    noisy = noisy.squeeze(0)
    # Add fake batch dimension and relative length tensor
    enhanced = enhance_model.enhance_batch(noisy.unsqueeze(0), lengths=torch.tensor([1.]))
    return noisy.numpy(), enhanced[0].cpu().numpy(), rate

# Main function
def main():
    # Set page title, icon, and layout
    st.set_page_config(page_title="Speech Enhancement | MetricGan+", page_icon="🔊", layout="wide")
    # Set main title
    st.title("Speech Enhancement - SpeechBrain - MetricGan+")

    # Display file uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=ALLOWED_EXTENSIONS)

    # If file is uploaded
    if uploaded_file is not None:
        # Check if file type is allowed
        if allowed_file(uploaded_file.name):
            # Process uploaded file
            with st.spinner("Processing..."):
                speech, enhanced, sr = process_file(uploaded_file)
            # Display original audio
            st.text("Original audio")
            st.audio(speech, format='audio/wav', start_time=0, sample_rate=sr)
            # Display enhanced audio
            st.text("Enhanced audio")
            st.audio(enhanced, format='audio/wav', start_time=0, sample_rate=sr)
        else:
            # Display warning if file type is not allowed
            st.warning("Invalid file type. Please upload a WAV file.")

if __name__ == '__main__':
    main()
