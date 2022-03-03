# DALI Lab Data Challenge 22W by Brian Wang

## Part 2 

### Objective 
Personally, I am very passionate about proper diagnosis and treatment for mental health illnesses. Currently, the most common way to assess anxiety is through point-in-time assessments like the GAD-7. Unfortunately, this survey and others like it can be subject to recall bias and do not fully capture the variability in an individual’s day-to-day experience. In this challenge, I aim to explore deep learning derived latent representations of daily anxiety that can capture a person’s varied daily experience better than point-in-time assessments. 

### The Datasets
This challenged used anxiety data collected from 30 teenagers over the course of a year, of which 12 met the standards for having clinical GAD based on at least one monthly GAD-7 assessment. This data was collected as a part of a study published in the Clinical Psychological Science journal, “A year in the social life of a teenager: Within-person fluctuations in stress, phone communication, and anxiety and depression.” Anxiety was measured at the end of every month using the GAD-7 (in `Monthly_Data.csv`) and three times each day with a one item anxiety severity assessment (in `Momentary_Data.csv`).

### Assumptions


### Data Preprocessing 
All preprocessing of this data was completed in Python and Pandas. It can be found in `preprocessing.ipynb` (`preprocessing.html` for a browser view of the notebook). The output can be found in `Formatted_Data.csv`. 

##### Separating data into person-months
To assess how an individual’s day-to-day experiences with anxiety relates to monthly GAD-7 assessments of their anxiety, each participant’s monthly GAD-7 scores and intradaily self-reported assessments of their anxiety level collected in the morning, afternoon, and evening were included in the present analysis. Participant's anxiety information was treated separately on a month-by-month basis, henceforth referred to as a person-month (e.g. the analysis treats data from month one of the study for person A as separate from data from month 4 of the study for person A). For each person-month, the monthly GAD-7 score and the 28 days of daily anxiety scores leading up to and inclusive of the day of their monthly in-person assessment is extracted. Thus, for each person-month, 84 points of daily anxiety data are considered due to intradaily collection of morning, afternoon, and evening measurements of their anxiety for 28 days. By processing the raw data through this method, 236 person-months worth of anxiety data are constructed. This breakdown of individuals into different person-months allows us to create more "individuals" in our study and allows us to focus on how a monthly measurement (GAD-7) relates to daily anxiety. 

Upon initial inspection of plots of monthly GAD scores vs. daily anxiety data, it seems like daily anxiety generally varies around the value for monthly GAD-7 scores. Yet GAD-7 scores are often prone to recency bias (affected most by the daily anxiety measurements in the days leading up to the monthly measurement) and fail to tell us the story about how variable a person's anxiety experience is. 

<img width="782" alt="Screen Shot 2022-03-03 at 3 53 18 PM" src="https://user-images.githubusercontent.com/62949093/156651143-287b18c7-ec0a-41d5-96e6-a5b9824c0a21.png">

##### Normalization
Daily anxiety measurements and monthly GAD-7 scores were normalized to be on a consistent scale (0-1). Similarly, missing values for anxiety measurements were assigned to a value of -1. Because daily anxiety measurements range from 1-7 and GAD-7 Scores range from 0-21, there are no significant outliers in terms of values in our dataset. 

##### Dealing with Outliers 
There was lots of missing daily anxiety measurements data. Person-months with over 70% of daily anxiety measurements missing were excluded from analysis, resulting in 66 of the original 236 person-months remaining eligible for analysis. Every participant in the study was represented by at least one person-month, with a maximum of three person-months for a given participant. 

### Model Building and Training 
All model building was completed in Tensorflow. It can be found in `modelling.ipynb` (`modelling.html` for a browser view of the notebook). The output can be found in `Formatted_Data.csv`. 

##### Model Architecture 
The first goal of the machine learning approach to analyzing the data is to see if the original feature space of the daily anxiety data (84 anxiety measurements per month) can be reduced into a smaller latent feature space that still represents all the variation in the original data. To accomplish this and ensure that the features were capturing enough variation, the model is built with an encoder-decoder framework, which interrogates if a smaller feature space can be decoded to roughly resemble the original sequence of daily anxiety data, thus suggesting that the features in the smaller feature space are capturing the variability in the larger feature space and serve as a comprehensive overview of a person’s day-to-day anxiety experience over the course of a month. 

To accomplish this goal an encoder-decoder neural network framework was built consisting mainly of Long-Short Term Memory (LSTM) nodes. Inputs to the LSTM model are 66 person-month’s worth of anxiety data separated into three features (morning, afternoon and evening anxiety) and 28 time points. The encoder section of the model is responsible for reducing this feature space into three features per person. The decoder portion of the model then reconstructs the daily anxiety data back from these three features. In the LSTM model, there are three layers for the encoder and three layers for the decoder. The encoder contained three layers of 64, 32 and 3 nodes, respectively, which was mirrored in the decoder framework. The final layer is a TimeDistributed layer wrapped on a Dense Layer on the output layer of the model. The dense layer has three units, which allows us to predict the sequence of anxiety data three measurements (morning, afternoon, evening) at a time (i.e. one day at a time). The model uses the ADAM optimization algorithm for training and uses the mean absolute error as the loss function. 

##### Model Tuning 
The model was trained with 46 person-months worth of data and was tested against 20 person-months worth of data. 
Many different layer sizes were tested and layer sizes were optimized for the encoder model (with the decoder model always having the same layer sizes as the encoder model). To measure the accuracy of each model’s decoding of the feature layer back to the original sequence, each time point was represented as having an x-value of its respective anxiety value in the original sequence and a y-value of its respective anxiety value in the predicted sequence. While building the model architecture, optimizing for a small feature layer size had to be balanced against the ability to decode this feature layer into the original sequence of daily anxiety data somewhat accurately. 
A 70%-30% train-test split for the model tuning framework. The model was trained for 1500 epochs and used the model weights for the epoch at which the testing and training loss began to diverge (as seen below in the training vs. validation curve). The latent feature space for the 66 person-months can be found in `output_LSTM_3_layer.csv`. 

<img width="666" alt="Screen Shot 2022-03-03 at 3 50 54 PM" src="https://user-images.githubusercontent.com/62949093/156650832-519d38d4-fc8e-4429-a5c1-e372aa4fce9d.png">

### Model Metrics and Analysis 
Model analysis primarily relied on Hierarchical Clustering, UMAP, and regression in R. Some preliminary analysis can be found in `modelling.ipynb` (`modelling.html` for a browser view of the notebook) and the remaining analysis can be found in `analysis.ipynb` (`analysis.html` for a browser view of the notebook). The output can be found in `Formatted_Data.csv`. 

##### r-values 
The analysis indicates that the LSTM model is quite reliable in predicting a sequence of a person’s anxiety data over the course of a month using only 3 features. We compute the r-values of a person’s actual sequence of anxiety over the course of a month versus the model’s predicted sequence of anxiety over the course of the month. With an average r-value of 0.8319 for the individuals in the testing data, it is evident that the 3 features extracted from the LSTM model are able to capture a person’s experience with anxiety over the course of a month quite well.

##### Regression Descriptors 

Because the UMAP analysis does not allow us to understand how the latent features are being determined, I used R to regress various properties of the daily anxiety data (variability, percent missing, etc.) against the 3 latent features. As shown below, feature 1 is heavily correlated with the mean daily anxiety (p-value of 0.00228) and feature 3 is heavily correlated with a measure of variance- root mean square of successive differences, which tells us roughly how much anxiety varied on a day-to-day basis for each person-month (p-value of 0.0147). Thus, it seems like our LSTM model is learning on both the mean anxiety and variance of anxiety to be able to properly reconstruct daily anxiety sequences (with 84 data points) from 3 latent features. 

<img width="740" alt="Screen Shot 2022-03-03 at 4 10 06 PM" src="https://user-images.githubusercontent.com/62949093/156653421-3534517b-3c7d-411e-9451-934c3a80bd06.png">
<img width="739" alt="Screen Shot 2022-03-03 at 4 10 12 PM" src="https://user-images.githubusercontent.com/62949093/156653423-ddbfc029-82c9-43e5-8329-254d4f955a17.png">
<img width="750" alt="Screen Shot 2022-03-03 at 4 10 16 PM" src="https://user-images.githubusercontent.com/62949093/156653424-9919489d-399a-48cf-8efc-ea3dd5cacaab.png">


