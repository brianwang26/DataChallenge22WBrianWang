# DALI Lab Data Challenge 22W by Brian Wang

## Part 2 

### Objective 
Personally, I am very passionate about proper diagnosis and treatment for mental health illnesses. Currently, the most common way to assess anxiety is through point-in-time assessments like the GAD-7. Unfortunately, this survey and others like it can be subject to recall bias and do not fully capture the variability in an individual’s day-to-day experience. In this challenge, I aim to explore deep learning derived latent representations of daily anxiety that can capture a person’s varied daily experience better than point-in-time assessments. 

### The Datasets
This challenged used anxiety data collected from 30 teenagers over the course of a year, of which 12 met the standards for having clinical GAD based on at least one monthly GAD-7 assessment. This data was collected as a part of a study published in the Clinical Psychological Science journal, “A year in the social life of a teenager: Within-person fluctuations in stress, phone communication, and anxiety and depression.” Anxiety was measured at the end of every month using the GAD-7 (in `Monthly_Data.csv`) and three times each day with a one item anxiety severity assessment (in `Momentary_Data.csv`).

### Data Preprocessing 
All preprocessing of this data was completed in Python. It can be found in `preprocessing.ipynb` (`preprocessing.html` for a browser view of the notebook). The output can be found in `Formatted_Data.csv`. 

##### Separating data into person-months
To assess how an individual’s day-to-day experiences with anxiety relates to monthly GAD-7 assessments of their anxiety, each participant’s monthly GAD-7 scores and intradaily self-reported assessments of their anxiety level collected in the morning, afternoon, and evening were included in the present analysis. Participant's anxiety information was treated separately on a month-by-month basis, henceforth referred to as a person-month (e.g. the analysis treats data from month one of the study for person A as separate from data from month 4 of the study for person A). For each person-month, the monthly GAD-7 score and the 28 days of daily anxiety scores leading up to and inclusive of the day of their monthly in-person assessment is extracted. Thus, for each person-month, 84 points of daily anxiety data are considered due to intradaily collection of morning, afternoon, and evening measurements of their anxiety for 28 days. By processing the raw data through this method, 236 person-months worth of anxiety data are constructed. This breakdown of individuals into different person-months allows us to create more "individuals" in our study and allows us to focus on how a monthly measurement (GAD-7) relates to daily anxiety. 

##### Normalization
Daily anxiety measurements and monthly GAD-7 scores were normalized to be on a consistent scale (0 - 1). Similarly, missing values for anxiety measurements were assigned to a value of -1. Because daily anxiety measurements range from 1-7 and GAD-7 Scores range from 0-21, there are no significant outliers in terms of values in our dataset. 

##### Dealing with Outliers 
There was lots of missing daily anxiety measurements data. Person-months with over 70% of daily anxiety measurements missing were excluded from analysis, resulting in 66 of the original 236 person-months remaining eligible for analysis. Every participant in the study was represented by at least one person-month, with a maximum of three person-months for a given participant. 





