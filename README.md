# Speaker-Diarization
Speaker Diarization is the procees which aims to find who spoke when in an audio and total number of speakers in an audio recording.
This project contains:
- Voice Activity Detection (webrtcvad)
- Speaker Segmentation based on Bi-LSTM
- Embedding Extraction (d-vector extraction)
- Clustering (k-MEANS and Mean Shift)
## Dataset
- Training Data\
To Train the Segmentation model Hindi Dataset was prepared.\The Link below contains 3 Audio files of Hindi News Channel Debate taken from Youtbue Video https://www.youtube.com/watch?v=fGEWWAly_-0 and Annotation file.\
The audio files were manually annotated with the help of Audacity Software. Annotation Format (Speaker Id,Offset, duration)\
[Link to Training Data Audio Files](https://drive.google.com/drive/folders/1jvSxEaMNx7IjzQIlrT4Vnl4x8TZTtZaB)
- Testing Data\
The Testing Data contains 4 Audio files and 1 Annotation file. The Annotation file is for just 3 Audio files (Hindi_01.wav , Hindi_02.wav and Hindi_03.wav).\
Desh.wav file is not manually annotated but can be used for testing the Diarization System. It's taken from Yotube Video https://www.youtube.com/watch?v=kqA9ISVcPD0&t=24s \
[Link to Testing Data](https://drive.google.com/open?id=16XCqfCaNo9djdx_TVK3hHxP6by3RaKU5)
## How to Run the code
For running the code in google collab you need to upload the required audio test file and wieghts of pre-trained model to your google drive account.
1. Mount the Google Drive.\
```
from google.colab import drive
drive.mount('/content/drive')
```
2. Define the path to pre-trained model file and testing audio file. You can add the files to your drive and redefine the path.

## Analysis
## References
