6ESTATES Project Test
=============
This test is to verify the interviewee's NLP & project capacity. 

## Task Description
This task is a traditional NLP task: Text Classification for Sentiment Analysis. The interviewee should show his/her ability to 

- process raw data to extract features.
- build one or more model (e.g. Logistic Regression, SVM, Decision Tree, Neural Network) to train on such data set
- test trained model on some evaluation metric (e.g. Precision, Recall, AUC)

Any tools or libraries can be used to finish this task.

## Data Set
the data set includes 4 files:

- train.json: the training set
- dev.json: the validation set if needed
- test.json: the test set 
- glove.840B.300d.txt: an open source word embedding if needed

Each line in json file is a single case. The 'sentence' field is the text and the 'label' field is the classification label.

## Submit Requirement
The interviewee should submit his/her codes with a single entry (e.g. run.sh, run.py) to train and test model.

The output should include

- prediction result for test.json
- evaluation result for training and predicting performance

A readme is required to briefly introduce the implementation and to list the tools and libraries used.


