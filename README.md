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


We performed total 4 experiments on 3 datasets.
1. AMI Corpus - As part of initial experimentation we produced results which were evaluated on DER mainly. 
   DER on AMI corpus was 35.9%.
![Capture1](https://user-images.githubusercontent.com/44304305/83499265-b681b100-a4da-11ea-9c7d-f5d59c8f5022.JPG)

2. Hindi A - To experiment with Hindi language we made this data set from a group discussion on youutube. This was mostly noisy and overalap was also more. Because of the noise and the overlap we got DER as 60%.
![Capture2](https://user-images.githubusercontent.com/44304305/83498771-f5fbcd80-a4d9-11ea-86eb-99ffa77a41b3.JPG)

3. Hindi B - The results from Hindi A were not convincing so we made another dataset we called Hindi B which had lesser overlaps and minimum noise. The DER we got was 12%.
![Capture3](https://user-images.githubusercontent.com/44304305/83498827-0ad86100-a4da-11ea-88de-03de63554460.JPG)


4. Semi supervised - To extend the approach we used semi supervised methodology where we trained our data on Hindi B and then tested on Hindi A. We got 27% DER.
![Capture4](https://user-images.githubusercontent.com/44304305/83499125-79b5ba00-a4da-11ea-91d0-bd420a9db610.JPG)





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
 

