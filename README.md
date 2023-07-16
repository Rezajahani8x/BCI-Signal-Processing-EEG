# BCI-Signal-Processing-EEG
In this project, it is aimed to develop a model to classify the action that a subject wants to take via EEG signal obtained from brain.

BCI competition is an event hold in National Brain Mapping Laboratory in University of Tehran.
In its second competition, EEG signals were obtained from different subjects doing one of the four below actions. This experiment was repeated several times for each action.

1- Finger Movement
2- Arm Movememnt
3- Foot Movement
4- No Action

The classifier is formed based on 3 sub-classifiers.
Classifier 1: Classifies the data between class 4 and class{1,2,3}
Classifier 2: Classifies the data between class 3 and class{1,2}
Classifier 3: Classifies the data between class 1 and class 2

Each classifier is developed according to CSP filters which mapps the data which is a multi-channel observation in T samples to a feature vector. Furthermore, an LDA vector is calculated to separate the classes.

For each subject butterworth filters with different parameters are applied to the data for enhancing the performance.
