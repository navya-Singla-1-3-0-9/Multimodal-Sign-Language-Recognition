# Feature Fusion and Multi-Stream CNNs for Scale-Adaptive Multimodal Sign Language Recognition

### Background: Sign Language Recognition
- Sign languages are expressed using a combination of hand movements and non-verbal elements, such as body language, and can be considered natural languages that have their own grammar and lexicon.
- Traditional approaches for sign language recognition involve capturing images or videos of sign language gestures and analyzing them to recognize the intended message. Some popular approaches include Dynamic Time Warping, Support Vector Machines. and Hidden Markov Models. 
- More recently, in  the domain of static sign language recognition (SLR), multimodal data inputs are being used to capture information across modalities and enhance accuracy.
<img width="771" alt="image" src="https://github.com/user-attachments/assets/52bb1b8c-877d-4204-8a1b-63bd5361388a">



### Background: Continuous and Sensor-based Sign Language Recognition
- Continuous SLR is a branch of SLR, which helps in sequentially translating image streams into a series of glosses to express a complete sentence. It consists of a visual, sequential and alignment module.
- There has been an increase in interest in using sensor data for SLR due to the accessibility of inexpensive sensors and technological breakthroughs in machine learning algorithms. Sensor based SLR systems use sensors attached to the body, such as accelerometers and gyroscopes, to track the movement of the user's body and recognize sign language gestures.

<img width="570" alt="image" src="https://github.com/user-attachments/assets/4a9a8675-bb05-4736-a612-e4aa25e62c35">

### CHALLENGES IN SIGN LANGUAGE RECOGNITION
The use of sign language is often limited by the lack of technology capable of recognizing and interpreting it. Within the domain of hand gesture analysis, a plethora of obstacles exist, including but not limited to:

- Occlusion
- challenges with feature extraction
- variations in hand size
- fluctuating backgrounds
- alterations in scale and orientation
- variations in illumination

### Major Contributions of this work
- **Feature Extraction** - A method for feature extraction using a fusion of the Gabor filter and LBP has been proposed for efficient extraction of hand shape and texture.
- **Multimodality** - By incorporating multiple streams through multi-stream CNNs, the network can improve its performance. The RGB stream learns color and texture, the depth stream learns shape and depth, and the feature-extracted RGB stream learns texture and spatial frequency.
- **Scale Variations & Dimensionality Reduction**: To improve the CNN architecture, SPP has been introduced. This technique allows the network to learn features at different scales, which helps it handle variations in object size and position, especially in hand gesture classification. Additionally, it can reduce the input dimensionality and focus on critical features at each 
level.

### Datasets Used
The method was evaluated on 4 datasets - the ASL Fingerspelling Dataset, the Massey University Gesture Dataset, the Indian Sign Language Dataset, and OpenGesture3D (RGB and Depth) Dataset.

<img width="490" alt="image" src="https://github.com/user-attachments/assets/abcf98a4-7ea6-4cb2-be28-04a1f92a2f1d">

### Methodology
<img width="635" alt="image" src="https://github.com/user-attachments/assets/ccfc36c4-c8a3-4ee6-904f-e89ad01a2332">
<img width="269" alt="image" src="https://github.com/user-attachments/assets/c4f32fa4-4e68-4b6f-ac1f-170760c0b602">

#### Multi-Stream CNNs
- In this work, we use a multi-stream CNN based approach for the classification of static sign language. 
- Three distinct inputs were utilized in the model as the streams of the CNN - a depth image, an RGB image, and a set of features obtained from the fusion of the Gabor filter and LBP techniques.
-   **Gabor Filter**: A 2-dimensional Gabor filter bank is one of the methods in our approach, utilized to convert images into their frequency representations
  <img width="312" alt="image" src="https://github.com/user-attachments/assets/631de490-36f2-4d7c-a1d4-3f6f6baef302">
-   **Local Binary Patterns**: LBP is a technique used to obtain information about texture from the images. It represents the pattern of intensity changes in an image by evaluating the pixel intensity values in relation to its neighboring pixels.
<img width="318" alt="image" src="https://github.com/user-attachments/assets/9bad31e7-f931-4c86-a556-f47734e19d3b">


#### Spatial Pyramid Pooling
- In order to increase the discriminative power of the proposed multi-stream CNN, a SPP layer was added after each stream. 
- By learning features at multiple scales, the network can better handle variations in object size and position within the image.
- This can be especially useful for hand gesture classification as hand gestures can vary greatly in size and position depending on the distance of the hand to the camera.

### Model Training and Evaluation Results
<img width="931" alt="image" src="https://github.com/user-attachments/assets/f38a8397-bb0d-444b-a91d-96b65f4d0de0">
<img width="948" alt="image" src="https://github.com/user-attachments/assets/a5fd4680-a9d3-4b56-8c7f-b2bbba7d8343">
<img width="930" alt="image" src="https://github.com/user-attachments/assets/2ff82304-ba15-43af-9b1b-8e5671dfe2b1">
<img width="949" alt="image" src="https://github.com/user-attachments/assets/6ea18885-2cce-4cb3-a01c-d913f5a1ec2c">

#### Comparison with different Feature Extraction Approaches
<img width="629" alt="image" src="https://github.com/user-attachments/assets/b644ae94-db94-4102-ac3f-42bcc546cfd8">
<img width="613" alt="image" src="https://github.com/user-attachments/assets/f55ae403-0938-485a-836a-78e4529700b5">

#### Performance Analysis
<img width="567" alt="image" src="https://github.com/user-attachments/assets/14ac0a2f-836b-4bcd-9866-0eab3d53895f">

#### Comparison with existing Works
<img width="918" alt="image" src="https://github.com/user-attachments/assets/da36756c-51cb-4023-8b94-85601ed2483f">


### Conclusion
- The presented work utilizes the multi-stream CNN architecture which is capable of learning features at multiple scales through the use of SPP.
- The use of multiple modalities including Color, depth information, and extracted image features provides complementary information for sign language recognition
- The feature fusion method based on the Gabor Filter and LBP provides extensive details about image texture and handshape, further improving recognition accuracy. 
- The performance of this method was contrasted with that of other feature extraction techniques - SIFT, Gabor filter, and LBP, and showed a higher accuracy rate in recognizing individual signs. 
- The method was evaluated on 4 datasets - the ASL Fingerspelling Dataset, the Massey University ASL Dataset, the Indian Sign Language Dataset, and OpenGesture3D Dataset, where it achieved accuracies of 99.6%, 99.67%, 99.93%, and 93.75% respectively.









