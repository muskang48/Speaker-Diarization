# Speaker-Diarization
Speaker Diarization is the procees which aims to find who spoke when in an audio and total number of speakers in an audio recording.
This project contains:
- Voice Activity Detection (webrtcvad)
- Speaker Segmentation based on Bi-LSTM
- Embedding Extraction (d-vector extraction)
- Clustering (k-MEANS and Mean Shift)
## Datset
- Training Data
To Train the Segmentation model Hindi Dataset was prepared. The Link below contains 3 Audio files of Hindi News Channel Debate taken from Youtbue Video https://www.youtube.com/watch?v=fGEWWAly_-0 and Annotation file.
The audio files were manually annotated with the help of Audacity Software. Annotation Format (Speaker Id,Offset, duration)
[Link to Training Data Audio Files](https://drive.google.com/drive/folders/1jvSxEaMNx7IjzQIlrT4Vnl4x8TZTtZaB)
- Testing Data
[Link to Testing Data](https://drive.google.com/open?id=16XCqfCaNo9djdx_TVK3hHxP6by3RaKU5)
