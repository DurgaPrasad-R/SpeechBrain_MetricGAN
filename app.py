import streamlit as st
import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement

ALLOWED_EXTENSIONS = {'wav'}

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank",
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(file):
    noisy, rate = torchaudio.load(file)
    try:
        assert rate == 16000, "sampling rate must be 16000"
    except AssertionError as e:
        return None, None, None, str(e)
    # Remove extra dimension from input signal
    noisy = noisy.squeeze(0)
    # Add fake batch dimension and relative length tensor
    enhanced = enhance_model.enhance_batch(noisy.unsqueeze(0), lengths=torch.tensor([1.]))
    return noisy.numpy(), enhanced[0].cpu().numpy(), rate, None

def main():
    st.set_page_config(page_title="Speech Enhancement | MetricGan+", page_icon="🔊", layout="wide")

    st.title("Speech Enhancement - SpeechBrain - MetricGan+")

    uploaded_file = st.file_uploader("Upload an audio file", type=ALLOWED_EXTENSIONS)

    if uploaded_file is not None:
        if allowed_file(uploaded_file.name):
            with st.spinner("Processing..."):
                speech, enhanced, sr, error = process_file(uploaded_file)
            if error is not None:
                st.error(error)
            else:
                st.text("Original audio")
                st.audio(speech, format='audio/wav', start_time=0, sample_rate=sr)
                st.text("Enhanced audio")
                st.audio(enhanced, format='audio/wav', start_time=0, sample_rate=sr)
        else:
            st.warning("Invalid file type. Please upload a WAV file.")

if __name__ == '__main__':
    main()
