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
The Testing Data contains 4 Audio files and 1 Annotation file. The annotation file is a csv file which was manually annotated and contains filename(Hindi_01/Hindi_02/Hindi_03) , Offset, Duration,  and Speaker Id. 
1. The Annotation file is for just 3 Audio files (Hindi_01.wav , Hindi_02.wav and Hindi_03.wav).Hindi_01, Hindi_02 and Hindi_03 are splitted from the audio recording taken from Hindi Yotubue Video of duration 1 hour 43 minutes.Link for the youtube recording - https://www.youtube.com/watch?v=1Yj8K2ZHttA&t=424s
2. Desh.wav file is not manually annotated but can be used for testing the Diarization System. It's taken from Yotube Video https://www.youtube.com/watch?v=kqA9ISVcPD0&t=24s. The duration of this Audio is approx 28 minutes \
[Link to Testing Data](https://drive.google.com/open?id=16XCqfCaNo9djdx_TVK3hHxP6by3RaKU5). This link also contains the annotated .csv file. for the (Hindi_01,Hindi_02,Hindi_03) audio files.
 - NOTE:
1. Almost all the files are near about duration of 30 minutes. If the duration is more than that it is split into the parts for making the duration less.\
2. The test audio files are completly unseen and no speaker is same in the training and testing dataset.
3. The .wav audio files was extracted from Youtube Video using (MiniTool uTube Downloader) Software. After extracting the .wav file it was converted from stereo to mono type using Audacity Software.Export the .wav file from the Software.The audio then have then specification (Sampling Rate:48000Hz, 16 bit PCM) 
## How to Run the code
This project contains 4 .ipynb files. One can open the files direclty in google Colab.\
Change_detection.ipynb file creates the model and train the model for Segmentation. sp_diarization.ipynb is the major file for Complete diarization and uses the saved pre-trained model.\
Segmentation.ipynb and VAD.ipynb are colab files to get seperate results for Segmentation and Voice Activity Detection.\
The complete dizrization system was evaluated for two clustering approaches kmeans and meanshift.kmeans.py and meanshift. .py files of both the clustering methods is uploaded.
- For running the code in google colab you need to upload the required audio test file and wieghts of pre-trained model to your google drive account.
1. Mount the Google Drive.
```
from google.colab import drive
drive.mount('/content/drive')
```
2. Define the path to pre-trained model file and testing audio file. You can add the files to your drive and redefine the path.
```
h5_model_file = '/content/drive/My Drive/SRU/model_hindi_2.h5'
segmented, n_clusters, hyp_df, result_hypo = diarization('/content/drive/My Drive/SRU/Hindi_01.wav')
reference, ref_df = reference_gen('/content/drive/My Drive/SRU/hindi_annotations1.csv')
```
## Analysis
Results-Testing the Model for Hindi_01.wav file having 7 Speakers. Duration of Audio-file (30 minutes 23 seconds)
1. Segmentation Model
```
TensorFlow 1.x selected.
Using TensorFlow backend.
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional_1 (Bidirection (None, 137, 256)          167936    
_________________________________________________________________
bidirectional_2 (Bidirection (None, 137, 256)          394240    
_________________________________________________________________
time_distributed_1 (TimeDist (None, 137, 32)           8224      
_________________________________________________________________
time_distributed_2 (TimeDist (None, 137, 32)           1056      
_________________________________________________________________
time_distributed_3 (TimeDist (None, 137, 1)            33        
=================================================================
Total params: 571,489
Trainable params: 571,489
Non-trainable params: 0
_________________________________________________________________
``` 
2. Clustering
   - Number of Speakers in an audio is equal to the number of clusters formed.
![Clusters](https://user-images.githubusercontent.com/61666843/80796608-415e4b80-8bbd-11ea-8eab-c15e5508d25b.png)

3. Segmentation

![Segmentation Results](https://user-images.githubusercontent.com/61666843/80796726-94d09980-8bbd-11ea-94f9-a952e55d9991.png)

4. Diarization Output Visulaiztion
    - Hypothesis\
    It shows who spoke when in an audio. 
    ![Hypothesis](https://user-images.githubusercontent.com/61666843/80796883-ff81d500-8bbd-11ea-8f16-313c674d9137.png)
    - Groundtrurh\
    It is the visulaization of manually annotated audio file.
    ![GroundTruth](https://user-images.githubusercontent.com/61666843/80796988-3f48bc80-8bbe-11ea-9b22-bce43b76b3ae.png)
 - Diarization Error Rate \
 DER - 26.7% (Using Mean-Shift Clustering) \
 DER - 33.8% (Using Kmeans Clustering)

## References 
 - https://pdfs.semanticscholar.org/edff/b62b32ffcc2b5cc846e26375cb300fac9ecc.pdf
 - https://github.com/pyannote/pyannote-audio 
 - https://arxiv.org/abs/1710.10468
 - https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0411.PDF
 - https://scikit-learn.org/stable/modules/clustering.html
 - https://arxiv.org/abs/1710.10467
 - https://github.com/Janghyun1230/Speaker_Verification
 - https://github.com/yinruiqing/change_detection
 - https://pypi.org/project/webrtcvad-wheels/
 - https://github.com/wblgers/py_speech_seg/tree/master/BiLSTM
 

