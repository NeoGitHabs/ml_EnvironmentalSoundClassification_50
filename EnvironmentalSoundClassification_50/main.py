import io
import torch
import uvicorn
import torch.nn as nn
import soundfile as sf
import streamlit as st
import torch.nn.functional as F
from torchaudio import transforms
from fastapi import FastAPI, HTTPException, UploadFile, File



class SimpleCNN(nn.Module):
    def __init__(self,):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 50)
        )
    def forward(self, audio):
        audio = audio.unsqueeze(1)
        audio = self.first(audio)
        audio = self.second(audio)
        return audio


transform = transforms.MelSpectrogram(
    sample_rate=22050,
    n_mels=64
)

max_len = 500

# classes = ['airplane', 'breathing', 'brushing_teeth', 'can_opening', 'car_horn', 'cat', 'chainsaw', 'chirping_birds',
#             'church_bells', 'clapping', 'clock_alarm', 'clock_tick', 'coughing', 'cow', 'crackling_fire', 'crickets', 'crow',
#             'crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock', 'drinking_sipping', 'engine', 'fireworks', 'footsteps',
#             'frog', 'glass_breaking', 'hand_saw', 'helicopter', 'hen', 'insects', 'keyboard_typing', 'laughing', 'mouse_click',
#             'pig', 'pouring_water', 'rain', 'rooster', 'sea_waves', 'sheep', 'siren', 'sneezing', 'snoring', 'thunderstorm',
#             'toilet_flush', 'train', 'vacuum_cleaner', 'washing_machine', 'water_drops', 'wind']

audio_app = FastAPI(title='Environment sounds')
model = SimpleCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = torch.load('labels_environmental_sound_classification_50.pth', weights_only=False)
model.load_state_dict(torch.load('model_environmental_sound_classification_50.pth', map_location=device))
model.to(device)
model.eval()


def change_audio(waveform, sr):
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.from_numpy(waveform.T).float()  # (channels, samples)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 22050:
        resample = transforms.Resample(orig_freq=sr, new_freq=22050)
        waveform = resample(waveform)

    spec = transform(waveform)

    if spec.shape[-1] > max_len:
        spec = spec[..., :max_len]
    if spec.shape[-1] < max_len:
        spec = F.pad(spec, (0, max_len - spec.shape[-1]))

    return spec

@audio_app.post('/predict/')
async def predict_audio(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='File not found')

        wf, sr = sf.read(io.BytesIO(data), dtype='float32')
        wf = torch.from_numpy(wf).T if not isinstance(wf, torch.Tensor) else wf

        spec = change_audio(wf, sr)
        spec = spec.unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_ind = torch.argmax(y_pred, dim=1).item()
            pred_class = classes[pred_ind]

        return {f'Index: {pred_ind}, Sound type: {pred_class}'}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(audio_app, host='127.0.0.1', port=8000)


# st.title('Environment Sounds Classifier')
# st.text('Upload audio (.wav) to recognize sound')
#
# file = st.file_uploader('Upload a file', type=['wav'])
#
# if not file:
#     st.warning('Upload a file')
# else:
#     st.audio(file)
#     if st.button('Recognize'):
#         try:
#             data = file.read()
#
#             wf, sr = sf.read(io.BytesIO(data), dtype='float32')
#             wf = torch.from_numpy(wf.T).float()  # ← исправлено
#
#             spec = change_audio(wf, sr)
#             spec = spec.unsqueeze(0).to(device)
#
#             with torch.no_grad():
#                 y_pred = model(spec)
#                 pred_ind = torch.argmax(y_pred, dim=1).item()
#                 pred_class = classes[pred_ind]
#
#             st.success(f'Index: {pred_ind}, Sound type: {pred_class}')
#
#         except Exception as e:
#             st.warning(f'Error: {e}')
