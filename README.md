# Analysing Cricket: Shot Recognition & Similarity



https://github.com/RITIK-12/CricketShotClassification/assets/54806252/cd640022-e5e9-4412-832b-5171bf582bcc



<be>

### ✯ Introduction

<br>

* Cricket is a globally celebrated sport with profound economic impacts, involving billions in revenue and extensive fan engagement.
* AI-driven data analytics is rapidly transforming cricket, offering new avenues for player development and strategic planning.
* This project focuses on classifying cricket shots from videos into distinct categories and exploring the similarities between these shots.
* By leveraging these insights, players can enhance their skills, and coaches can identify promising new talent more effectively.


<br>

###  ✯ Dataset Preparation

<br>

<p align="center">
 <img width="900" height="400 alt="image" src="https://github.com/RITIK-12/CricketShotClassification/assets/54806252/f9fd9c8f-925c-41ec-957a-306bae1fbdb4">
</p>


* Dataset was sourced from the authors of CrickShot10 [[1]](https://www.researchgate.net/publication/350998665_CricShotClassify_An_Approach_to_Classifying_Batting_Shots_from_Cricket_Videos_Using_a_Convolutional_Neural_Network_and_Gated_Recurrent_Unit).
* Removed score texts from videos for better shot analysis.
* Applied horizontal flips to data for a better representation of different batting styles.
* Divided the dataset into training, validation, and testing sets with a 70-20-10 split for effective model evaluation.

<br>

### ✯ Model Architecture

<br>

<table>
<tr>
<td>
<img width="300" src="https://github.com/RITIK-12/CricketShotClassification/assets/54806252/ae57dcd3-7f0e-457a-830b-ef7d37f5df1f" />
</td>
<td>
  
<br>

* Utilises a CNN-based feature extractor from the EfficientNet family, applied in a time-distributed block to maintain temporal information across video frames.
* Incorporates a Global Average Pooling layer within the time-distributed block to condense features into a more manageable form.
* Employs GRU units to capture and analyse temporal dependencies between frames, enhancing the model's understanding of motion and sequence.
* Concludes with dense layers topped with a softmax activation for classifying shots into distinct categories based on learned features.

</td>
</tr>
</table>

<br>


### ✯  Model Training

<br>



| Model               | Training Accuracy | Validation Accuracy |
|---------------------|-------------------|---------------------|
| EfficientNetB0      | 100%              | 85.80%              |
| EfficientNetV2B0    | 100%              | 77.01%              |
| EfficientNetB4      | 100%              | 72.86%              |


* Built three model variants, each with a distinct feature extractor head to evaluate performance variations.
* Trained all models for 20 epochs using batch sizes of 16, processing 30 frames per video to capture temporal dynamics.
* Utilized the Adam optimizer, configured with a learning rate of 0.001, to efficiently converge to optimal weights.
* Employed sparse categorical crossentropy as the loss function, for handling class labels as integers.


<br>



### ✯  Optimising Performance with Genetic Algorithm-Based Hyperparameter Tuning


<br>



<img width="931" alt="image" src="https://github.com/RITIK-12/CricketShotClassification/assets/54806252/d347cff6-2515-4f12-9684-9e97415b7b2e">


* Each individual in the population represents a set of model hyperparameters, such as learning rate and epochs.
* Individuals are assessed based on the validation accuracy of the model trained with their hyperparameters.
* Randomly selects small groups of individuals, with the best-performing individual from each group chosen to continue to the next generation.
* Combines and modifies selected individuals' hyperparameters to explore new solutions and improve model performance.
* The stagnation_limit is set to 10, meaning the genetic algorithm halts if there's no improvement in the best fitness score after 10 consecutive generations. This mechanism efficiently conserves computational resources and prevents overfitting.
* The learning rate is between 0.0001 and 0.02, and the epochs range from 1 to 20, ensuring a comprehensive exploration of the hyperparameter space.


<br>



### ✯  Model Evaluation


<br>



| Model            | Testing Accuracy | Precision | Recall | F1 Score |
|------------------|------------------|-----------|--------|----------|
| EfficientNetB0   | 94%              | 94%       | 94%    | 94%      |
| EfficientNetV2B0 | 81%              | 82%       | 81%    | 81%      |
| EfficientNetB4   | 74%              | 75%       | 74%    | 74%      |


* All three models were evaluated on the test set.
* Accuracy, Precision, Recall, F1-score were the metrics used for evaluation.
* Model with EffiecientNet B0 backbone outperformed the other two models.


<br>


### ✯  Analysing and Assessing Cricket Shot Similarities


<br>

<p align="center">
<img width="256" alt="image" src="https://github.com/RITIK-12/CricketShotClassification/assets/54806252/befe21af-efc1-4b54-bfcf-3495c7800e2c">
<p>

* Extracted features from the convolutional block of the EfficientNet backbone, mapping them into a concise vector representation.
* Calculated cosine distance between feature vectors to assess similarities across different video inputs.
* Utilized this distance metric to determine the degree of similarity between two cricket shot videos.
* Confirmed model accuracy with a 100% similarity score for identical input videos, validating the effectiveness of the feature extraction and comparison approach.


<br>



### ✯  References


<br>


1. A. Sen, K. Deb, P. K. Dhar, and T. Koshiba, "CricShotClassify: An Approach to Classifying Batting Shots from Cricket Videos Using a Convolutional Neural Network and Gated Recurrent Unit," Sensors, vol. 21, no. 8, Art. no. 2846, 2021. [Online]. Available: https://doi.org/10.3390/s21082846.
2. M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in Proc. 36th Int. Conf. Mach. Learn., Long Beach, CA, USA, 2019, vol. 97, pp. 6105–6114. [Online]. Available: http://proceedings.mlr.press/v97/tan19a.html
3. K. Cho et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," arXiv preprint arXiv:1406.1078, 2014. [Online]. Available: https://arxiv.org/abs/1406.1078
4. M. Abadi et al., "TensorFlow: Large-scale machine learning on heterogeneous systems," 2016. [Online]. Software  available: https://www.tensorflow.org/
5. "Streamlit: The fastest way to build custom ML tools," Streamlit. Accessed: Apr. 17, 2024. [Online]. Available: https://www.streamlit.io/

