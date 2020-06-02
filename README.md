# Speaker-Diarization
Speaker Diarization is the procees which aims to find who spoke when in an audio and total number of speakers in an audio recording.
This project contains:
- Voice Activity Detection (webrtcvad)
- Speaker Segmentation based on Bi-LSTM
- Embedding Extraction (d-vector extraction)
- Clustering (k-MEANS and Mean Shift)
# Voice Activity Detection
Voice activity detection (VAD) is a technique in which the presence or absence of human speech is detected. This part has been completed using a module devloped by google called as WebRTC. It's an open framework for the web that enables Real-Time Communications (RTC) capabilities in the browser. The voice activity dector is one of the specific module present in WebRTC. This basic working of WebRTC based VAD is as,
- WebRTC VAD is a Gaussian Mixture Model(GMM) based voice activity detector 
- GMM model using PLP features
- Two full covariance Gaussians: One for speech, and one for Non-Speech is used.
To learn about PLP we followed this paper\
[link](http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2005/Hoenig05-RPL.pdf). \
Following is breif analysis of PLP. PLP consists of the following steps: 
(i) The power spectrum is computed from the windowed speech signal. \
(ii) A frequency warping into the Bark scale is applied. \
(iii) The auditorily warped spectrum is convoluted with the power spectrum of the simulated critical-band masking curve to simulate the critical-band integration of human hearing. \
(iv) The smoothed spectrum is down-sampled at intervals of ≈ 1 Bark. The three steps frequency warping, smoothing and sampling (ii-iv) are integrated into a single filter-bank called Bark filter-bank.\
(v) An equal-loudness pre-emphasis weights the filter-bank outputs to simulate the sensitivity of hearing.\
(vi) The equalized values are transformed according to the power law of Stevens by raising each to the power of 0.33. The resulting auditorily warped line spectrum is further processed by (vii) linear prediction (LP). Precisely speaking, applying LP to the auditorily warped line spectrum means that we compute the predictor coefficients of a (hypothetical) signal that has this warped spectrum as a power spectrum. \
Finally, (viii), cepstral coefficients are obtained from the predictor coefficients by a recursion that is equivalent to the logarithm of the model spectrum followed by an inverse Fourier transform. Following figure shows a comparative scheme of PLP computation. \
![01](https://user-images.githubusercontent.com/57112474/83497304-e7141b80-a4d7-11ea-91c8-5061103a15b0.JPG) \
So in WebRTC these PLP features are compyed for 10 ms frame and then two Variance models are comapred to find wheater this single frames audio or not.
# Speaker Segementation
Speaker segmentation constitues the heart of speaker diarization, the idea to exactly identify the location of speaker change ooint in the order to miliseconds is still an open chalenge. Speaker segmentation constitutes the heart of speaker diarization, the idea to exactly identify the location of speaker change point in the order of milliseconds is still an open challenge. For this part we have tried to develop a state of art system which is BiLSTM network that is trained using a special SMORMS3 optimizer. SMORMS3 optimizer is a hyvrid oprimizer devloped using RMSprop and Adam optimizers. SMORMS3 stands for "squared mean over root mean squared cubed". Following link provied a detailied analysis of SMORMS3 [Link](https://sifter.org/~simon/journal/20150420.html). \
Now, comming to our speaker segmentation part architecture which ultilizes this SMORMS3 optimizer wroks on the pricipal that address speaker change detection as a binary sequence labeling task using Bidirectional Long Short-Term Memory recurrent neural networks (Bi-LSTM). Given an audio recording, speaker change detection aims at finding the boundaries between speech turns of different speakers. In Figure below, the expected output of such a system would be the list of timestamps between spk1 & spk2, spk2 & spk1, and spk1 & spk4.
- First we extract the features, let x be a sequence of MFCC features extracted on a short (a few milliseconds) overlapping sliding window. The speaker change detection task is then turned into a binary sequence labeling task by defining y = (y1, y2...yT ) ∈ {0, 1}^T
such that y_{i} = 1 if there is a speaker change during the ith frame, and y_{i} = 0 otherwise. The objective is then to find a function f : X → Y that matches a feature sequence to a label sequence. We propose to model this function f as a recurrent neural network trained using the binary cross-entropy loss function.
![02](https://user-images.githubusercontent.com/57112474/83499789-85ee4700-a4db-11ea-8823-1874f8667004.JPG)
- The actual architecture of the network f is depicted in Figure below. It is composed of two Bi-LSTM (Bi-LSTM 1 and 2) and a multi-layer perceptron (MLP) whose weights are shared across the sequence. Bi-LSTMs allow to process sequences in forward and backward directions, making use of both past and future contexts. The output of both forward and backward LSTMs are concatenated and fed forward to the next layer. The shared MLP is made of three fully connected feedforward layers, using tanh activation function for the first two layers, and a sigmoid activation function for the last layer, in order to output a score between 0 and 1. \
![03](https://user-images.githubusercontent.com/57112474/83500189-12990500-a4dc-11ea-99b7-f30a1b393583.JPG)
# Combining VAD and Speaker Segmentation
In our code once the results from above modules were obtained, we combined them the results in logical way such that we had obtained frames of arbitrary seconds depending on the voiced part and the speaker change part. This can be explained with an example, lets us suppose we have first performed VAD and found that from 2 to 3 seconds there is some voice. In next part of speaker segmentation, we found that at 2.5 seconds there is a speaker change point. So, what we did we splited this audio frame of 1 seconds into two parts frame 1 from 2 to 2.5 seconds and then from 2.5 to 3. Similarly lets us suppose that we s=find that from 3 to 3.5 seconds there is some voice and then there is a silence of 1 seconds i.e. exactly at 4 sec some voice is coming into play. Now using Speaker change part we found that at 4 seconds there is speaker changepoint again we combined it in such way we defined there is new speaker at 4 seconds. All such logical results were combined to giver per frame output.
```
def group_intervals(a):
    a = a.tolist()
    ans = []

    curr = None
    for x in a:
        # no previous interval under consideration
        if curr == None:
          curr = x
        else:
            # check if we can merge the intervals
            if x[0]-curr[1] < 1:
                curr[1] = x[1]
            else:
            # if we cannot merge, push the current element to ans
                ans.append(curr)
                curr = x

        if curr is not None:
            ans.append(curr)

    d1 = np.asarray(ans)
    d2 = np.unique(d1)
    d3 = d2.reshape(int(len(d2)/2),2)
    return d3
    
def spliting(seg,arr):
  arr1 = arr.tolist()
  temp = arr.copy()
  
  for i in range(len(seg)-1):
    temp1 = float(seg[i])
    # print(temp1)
    for j in range(len(arr)-1):
      if ((temp1 > arr[j][0]) & (temp1 < arr[j][1])):
        arr1[j].insert(-1,(temp1))

  for i in range(len(arr1)-1):
    size=len(arr1[i])
    if size>=3:
      arr1[i].pop(-2) if arr1[i][-1]-arr1[i][-2]<0.2 else True
      
  return arr1
  
def final_reseg(arr):
  z=[]
  for i in arr:
    if len(i)==2:
      z.append(i)
    else:
      temp = len(i)
      for j in range(temp-1):
        if j!=temp-1:
          temp1 = [i[j],i[j+1]-0.01]
          z.append(temp1)
        elif j==temp-1:
          temp1 = [i[j],i[j+1]]
          z.append(temp1)
  
  return np.asarray(z)
```
# Embedding Extraction
This part now has to handle the idea to differentiate speakers. As mentioned in previous parts the frames extracted will go through the process of feature extraction. Let’s suppose we have a frame of 3 seconds starting from 4.5 to 7.5 sec, we extract the d-vectors for first 1.5 or 2 seconds of a single frame. To extract d-vectors we use the pyannote libraries pretrained models. The detailed analysis of pyannote can found uisng theor github repo [Link](https://pyannote.github.io/) and also from this [paper](https://arxiv.org/pdf/1911.01255.pdf).
# Clustering (k-MEANS and Mean Shift)
Clustering is one of the most common exploratory data analysis technique used to get an intuition about the structure of the data. It can be defined as the task of identifying subgroups in the data such that data points in the same subgroup (cluster) are very similar while data points in different clusters are very different. In other words, we try to find homogeneous subgroups within the data such that data points in each cluster are as similar as possible according to a similarity measure such as euclidean-based distance or correlation-based distance. The decision of which similarity measure to use is application-specific.
## Kmeans Algorithm
Kmeans algorithm is an iterative algorithm that tries to partition the dataset into Kpre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group. It tries to make the intra-cluster data points as similar as possible while also keeping the clusters as different (far) as possible. It assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster’s centroid (arithmetic mean of all the data points that belong to that cluster) is at the minimum. The less variation we have within clusters, the more homogeneous (similar) the data points are within the same cluster.
The way kmeans algorithm works is as follows:
- Specify number of clusters K.
- Initialize centroids by first shuffling the dataset and then randomly selecting K data points for the centroids without replacement.
- Keep iterating until there is no change to the centroids. i.e assignment of data points to clusters isn’t changing.
- Compute the sum of the squared distance between data points and all centroids.
- Assign each data point to the closest cluster (centroid).
- Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster.
The approach kmeans follows to solve the problem is called Expectation-Maximization.
## Mean Shift Algorithm
Mean Shift is very similar to the K-Means algorithm, except for one very important factor, you do not need to specify the number of groups prior to training. The Mean Shift algorithm finds clusters on its own. For this reason, it is even more of an "unsupervised" machine learning algorithm than K-Means. Mean shift builds upon the concept of kernel density estimation (KDE). KDE is a method to estimate the underlying distribution for a set of data. It works by placing a kernel on each point in the data set. A kernel is a fancy mathematical word for a weighting function. There are many different types of kernels, but the most popular one is the Gaussian kernel. Adding all of the individual kernels up generates a probability surface (e.g., density function). Depending on the kernel bandwidth parameter used, the resultant density function will vary. \
Mean shift exploits KDE idea by imagining what the points would do if they all climbed up hill to the nearest peak on the KDE surface. It does so by iteratively shifting each point uphill until it reaches a peak. Depending on the kernel bandwidth used, the KDE surface (and end clustering) will be different. As an extreme case, imagine that we use extremely tall skinny kernels (e.g., a small kernel bandwidth). The resultant KDE surface will have a peak for each point. This will result in each point being placed into its own cluster. On the other hand, imagine that we use an extremely short fat kernels (e.g., a large kernel bandwidth). This will result in a wide smooth KDE surface with one peak that all of the points will climb up to, resulting in one cluster. Kernels in between these two extremes will result in nicer clusterings. Below are two animations of mean shift running for different kernel bandwidth values.
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
 

