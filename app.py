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
    if rate != 16000:
        # Resample audio to 16KHz
        resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)
        noisy = resampler(noisy)
    assert noisy.shape[0] == 1, "input audio should have 1 channel"
    noisy = noisy.squeeze(0)
    # Convert the bit depth to 16 bits
    noisy = torchaudio.transforms.BitDepth(16)(noisy)
    # Add fake batch dimension and relative length tensor
    enhanced = enhance_model.enhance_batch(noisy.unsqueeze(0), lengths=torch.tensor([1.]))
    return noisy.numpy(), enhanced[0].cpu().numpy(), rate

def main():
    st.set_page_config(page_title="Speech Enhancement | MetricGan+", page_icon="🔊", layout="wide")

    st.title("Speech Enhancement - SpeechBrain - MetricGan+")

    uploaded_file = st.file_uploader("Upload an audio file", type=ALLOWED_EXTENSIONS)

    if uploaded_file is not None:
        if allowed_file(uploaded_file.name):
            with st.spinner("Processing..."):
                speech, enhanced, sr = process_file(uploaded_file)
            st.text("Original audio")
            st.audio(speech, format='audio/wav', start_time=0, sample_rate=sr)
            st.text("Enhanced audio")
            st.audio(enhanced, format='audio/wav', start_time=0, sample_rate=sr)
        else:
            st.warning("Invalid file type. Please upload a WAV file.")

if __name__ == '__main__':
    main()
