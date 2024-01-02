# Naive Bayes Algorithm

Naive Bayes is a probabilistic machine learning algorithm based on Bayes' theorem. The fundamental idea is to predict the probability of a particular instance belonging to a certain class given its feature values. The "naive" assumption in Naive Bayes is that the features are conditionally independent, given the class label. 

Bayes' theorem is expressed as:

\[ P(y|x) = \frac{P(x|y) \cdot P(y)}{P(x)} \]

where:
- \( P(y|x) \) is the posterior probability of class \(y\) given the features \(x\).
- \( P(x|y) \) is the likelihood, representing the probability of observing features \(x\) given class \(y\).
- \( P(y) \) is the prior probability of class \(y\).
- \( P(x) \) is the probability of observing the features \(x\), also known as the evidence.

The naive assumption implies that the features are conditionally independent, given the class, allowing us to express the likelihood as the product of individual feature probabilities:

\[ P(x|y) = P(x_1|y) \cdot P(x_2|y) \cdot \ldots \cdot P(x_n|y) \]

This simplifies the Bayes' theorem to:

\[ P(y|x) = \frac{P(x_1|y) \cdot P(x_2|y) \cdot \ldots \cdot P(x_n|y) \cdot P(y)}{P(x)} \]

The classifier assigns the class label that maximizes the posterior probability, making predictions based on these probabilities.

In the context of Gaussian Naive Bayes, where features are assumed to follow a Gaussian distribution, the likelihood term \( P(x_i|y) \) is modeled as a Gaussian distribution with mean \( \mu_{y,i} \) and variance \( \sigma_{y,i}^2 \). The parameters are estimated from the training data.

Despite its simplicity and the independence assumption, Naive Bayes is widely used due to its efficiency, especially in text classification and spam filtering, where it often performs well in practice.
