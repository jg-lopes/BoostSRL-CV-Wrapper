# BoostSRL Cross Validation Wrapper

## Introduction

This is an wrapper which enables the convenient usage of Stratified Cross Validation for BoostSRL models.

## How to use 

Download boostsrl.jar and auc.jar from the [StARlinG Lab website](https://starling.utdallas.edu/software/boostsrl/)

Certify that you have scikit-learn and numpy avaliable for import.

Set up the files in the data/ folder with your entire dataset (that is, there is no need for previous train and test split) and model. There is also no need to manually set the target. The system will detect which is the desired target from the positive and negative files.

Execute wrapper.py.

The python script will then separate all the examples in the data folder in the respective folds, which will be trained and tested separately. The means of all metrics obtained will then be printed to the console and written in a results.txt file.

## What is BoostSRL?

BoostSRL is a ["gradient-boosting based approach to learning different types of SRL models." ](https://starling.utdallas.edu/software/boostsrl/wiki/#) BoostSRL is an excellent resource for people interested in the area of relational machine learning. However, since it is located in a .jar file, which can only be interacted with via console commands and the alteration of the text files which feed data to the algorithm

This, of course, gets very tedious when doing work such as cross validation, which requires multiple executions of the program.

This wrapper serves to alleviate that concern, streamlining the overhead necessary to begin working effectively in training your models and measuring their results reliably.
