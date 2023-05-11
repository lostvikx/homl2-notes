# Notes: ML Landscape
A few questions the intrigued me on the first page of chapter one, were:
* Where does machine learning start and where does it end?
* What exactly does it mean for a machine to learn something?

# What is Machine Learning?
* Machine learning is the science (and art) of programming so that they can learn from data.
* Machine is the field of study that gives computers the ability to learn without being exlicitly programmed.

Machine Learning is great for:
* Problems for which existing solutions require a lot of fine-tuning or defining a long list of rules. Machine Learning algorithms can simplify code and perform better than traditional approaches.
* Complex problems for which traditional approach yields no good solution.
* Fluctuating environment: ML systems can adapt to new data.
* Getting insights about complex problems and large amounts of data: Data Mining.

Examples of ML Applications:
1. Detecting tumors in brain scans
2. Summarizing long documents automatically
3. Customer segmentation based on their purchases so that you can design different marketing strategies
4. Recommending an offering that a client may be interested in, based on past user activity
5. Building an intelligent bot for a game

# Types of Machine Learning Systems
## Supervised Learning
* The training set we feed the algorithm includes the desired solutions, called labels. 
* Typical tasks include classification and predicting a target numeric value by given set of features called predictors, this is called regression.
* Logistic regression is used for classification, as it outputs a value that corresponds to the probability of belonging to a given class.
* Algorithms: k-nearest, linear & logistic regression, SVMs, decision trees, random forests, neural networks.

## Unsupervised Learning
* The training dataset we feed the algorithm is unlabeled.
* Common tasks include clustering, anomaly detection, visualization, dimensionality reduction, and association rule learning.
* Dimensionality reduction: simplify the data without lossing too much information, makes the algorithm run faster, data will take up less space in memory and storage, and in some cases it may also yield better results.

## Semisupervised Learning
* Since labeling the data is a time consuming and costly, we may come across data with a few labeled instances.
* Most semisupervised learning algorithm are a combination of unsupervised and supervised algorithms: deep belief networks (DBNs).

## Reinforcement Learning
* The learning system is called an agent.
* An agent can observe the environment, select and perform actions and get rewards or penalities in return.
* The agent must learn by itself the best strategy, called policy.

# Challenges of Machine Learning
1. Insufficient quantity of data, machine learning algorithms need lots of data to work properly. Even to solve a simple problem, we need thousands examples. Data matters more than algorithms for complex problems, however small and medium sized datasets are still very common, and it is not always easy or cheap to get a lot of training data. So, selecting and developing algorithms is still a useful endeavor.
2. It is crutial that our training data be a representative of the new cases we want to generalize to. If data is too small, we have sampling noise. Even large samples can be non-representative if the sampling method is flawed, this is called sampling bias.
3. If the training data is full of errors, outliers, and noise, it will be harder for the system to detect the underlying patterns in the dataset.
4. Training data should contain enough relevant features and not too many irrelevent features. Feature engineering.
5. Overfitting and underfitting the training data.

# Exercise
1. Machine learning is the science of programming computer systems to provide outputs by learning from data instead of being programmed explicitly.
2. If a problem is too difficult to explicitly program a computer to perform the task, eg. handwriting & speech recognition, customer segmentation by clustering, and predicting using regression.
3. A labeled training set is a dataset that includes the desired solutions that we would like our ML algorithm to output.
4. Classification and regression.
5. Clustering, dimentionality reduction, anomaly detection, and visualization.
6. I will use a reinforcement learning methods to allow the robot to interact and learn in an environment, and come up with the optimal policy to walk in certain terrains.
7. Use of clustering ML algorithm is prudent to segment our customers.
8. Email spam detection can be classified as a supervised learning problem, as we need to give some clues (labels) to ascertain a spam email.
9. Online learning, we train the system incrementally by feeding it data instances sequentially.
10. Out of core learning is a type of online learning that is used to train a system on huge datasets that cannot fit one machine's memory, so the algorithm loads a part of the data, runs a training step on the part, and repeats this process.
11. An instance-based learning system learns the examples by heart, then generalizes to new cases by using similarity measure to compare the learned examples to new ones.
12. A model has one or more parameters to help it predict the outputs for new examples, eg. coefficient and slope in a linear model (weights and bias), a model tries to optimally compute the values of these parameters to generalize to new instances.
13. A model-based learning algorithm searches for optimal values for its model parameters, so that it can generalize well with new instances of data. Most common strategy used by a model is to minimize the cost function, it measures how bad the prediction is of a new instance. The features of the new instance is feed into the system, output is computed using the model parameters that were found in the learning step.
14. Insufficient quantity of data instances; data with errors, outliers, and noise; overfitting of the model; and lack of relevant features.
15. If our model is performing great on the training data, but generalizes poorly to new instances, we have a case of overfitting the training data. Three possible solutions: use a model with less parameters, remove outliers and errors present in the training data, and increasing the amount of training data, also removing any irrelevant attributes.
16. Before beginning the entire dataset must be divided into two sets, a training and a test set. A test set is used to evaluate the accuracy of our model, after being trained on the training set. The test set can help determine the generalization error rate.
17. A validation set is created when we hold out a part of the training set to evaluate several models with different hyperparameters on the reduced set. Then select the model that performs best on the validation set, then train the best model on the full training set, lastly we evaluate the generalization error on the test set. That way we can know the real generalization error of our model.
18. A train-dev set is a part of training of training set (just like validation set), it is held out to check if there is data mismatch between the training and validation and test sets.
19. If we tune the hyperparameters while evaluating the training set on the test set, we may only select the model that will be the best model for that particular set (test). The real world performance will be significantly worst, hence we don't tune hyperparameters on test set, but on the validation set.
