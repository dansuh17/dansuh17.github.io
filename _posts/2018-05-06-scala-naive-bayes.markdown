---
layout: post
title: "Functional Machine Learning - Training MNIST with simple Naive Bayes Classifier in Scala"
date: 2018-05-06 14:12:00 +0900
---

Few months into functional programming by learning Haskell, I've made a [capitalist choice](https://www.stackoverflowbusiness.com/blog/the-highest-paying-web-development-languages-in-2017) to learn scala. 
It didn't seem to me that scala isn't as beautiful and elegant as was told by my college professor,
but with love of functional programming, I've tried to implement a simple Naive Bayes Classifier, which I've also studied a while ago.

<!--more-->

## Abstraction of DataSet

Before we begin training anything, we first need an abstraction of the dataset.

- type `LabeledData`, representing a labeled data (supervised learning here)
- type `DataSet`, representing a data set, consisting of multiple `LabeledData`s along with some helper functions

### LabeledData

`LabeledData` is parameterized by two types - the type of label and the type of sample. 
It is simply a tuple of two elements `(label, sample)` (called `Product2` in scala), but with aliased accessors to `_1` and `_2` to increase readability.

```scala
trait LabeledData[LabelType, SampleType] extends Product2[LabelType, SampleType] {
  def label: LabelType = _1
  def sample: SampleType = _2
}

// companion object
object LabeledData {
  def apply[A, B](l: A, s: B): LabeledData[A, B] = new LabeledData[A, B] {
    override def _1: A = l
    override def _2: B = s
    override def canEqual(that: Any): Boolean = that.isInstanceOf[LabeledData[A, B]]
  }
}
```

### DataSet

`DataSet` is just a sequence of `LabeledData`s, but it should also offer some functions for easy processing later on.
Especially, `groupByClass` function, which groups samples according to their labels, will become very handy when we calculate means and variances.

Any derivatives of `DataSet` will contain `data` as its member value. It contains multiple `LabeledData` having labels of type `Label` and samples of type `Sample`.

```scala
trait DataSet[Label, Sample] {
  // type alias for sequence of labeled data
  type DataSeqType = Seq[LabeledData[Label, Sample]]

  // the actual data
  def data: Seq[LabeledData[Label, Sample]]
  def size: Int = data.size

  // distinct classes (labels) of dataset
  def classes: Seq[Label] = data.map(_.label).distinct

  // groups samples by their classes
  lazy val groupByClass: Map[Label, Seq[Sample]] =
    data groupBy(_.label) mapValues(_.map(_.sample))
}

object DataSet {
  // construct data from sequence of tuples
  def fromTupleSequence[A, B](rawData: Seq[(A, B)]): Seq[LabeledData[A, B]] =
    rawData.map(a => LabeledData(a._1, a._2))
}
```

## MnistDataSet

Now we extend `DataSet` to manipulate [MNIST dataset](http://yann.lecun.com/exdb/mnist/). 
MNIST consists of 60,000 training examples and 10,000 test examples. 
Each sample is a handwritten number of 28 x 28 pixels having labels 0 to 9, the actual number written.

![MNIST images](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)
*MNIST dataset - handwritten digits from 0 to 9*

The type for label is `Int` (the digit), and the type for each sample is a `Seq[Int]`. 
The length of each sample is 784 (28 x 28 pixels), whose values are 0 to 255.

So we extend `DataSet[Int, Seq[Float]]` to create `MnistDataSet`. 
Implementations of helper functions such as `mean` and `variance` are added to help calculate statistical values conveniently. 
These are implemented in the inherited class, not in the trait `DataSet`, because of the ambiguity of both `Label` and `Sample` types - what if `Sample` is a sequence of `Boolean`s? (Any functional advice on this?)
I don't think the implementations of both `mean` and `variance` is optimal, but it is, at least, straightforward.

```scala
case class MnistDataSet(data: Seq[LabeledData[Int, Seq[Float]]])
  extends DataSet[Int, Seq[Float]] {

  // find the mean of values
  def mean(list: Seq[Float]): Float = list.sum / list.size

  // find the varaiance of values
  def variance(list: Seq[Float], mean: Float): Float =
    (list.map(x => math.pow(x - mean, 2)).sum / list.size).toFloat

  // given a list, calculate both the mean and variance
  def meanVariance(list: Seq[Float]): (Float, Float) = {
    lazy val avg = mean(list)
    (avg, variance(list, mean = avg))
  }

  lazy val sampleMeanVariances: Map[Int, Seq[(Float, Float)]] =
    // here the values are sequence of Seq[Float], whose length is 784 (for MNIST)
    // transpose of this will make 784 sequences, grouping the samples by attributes
    groupByClass.mapValues(_.transpose.map(meanVariance).toVector)
}
```

We can now read in MNIST dataset from file using `DataSet`'s `fromTupleSequence` function, 
with the help of file reader I've borrowed from [swiftlearner](https://github.com/valdanylchuk/swiftlearner/blob/master/src/test/scala/com/danylchuk/swiftlearner/data/Mnist.scala).

```scala
val minstDataSet: MnistDataSet = MnistDataSet(DataSet fromTupleSequence labeledTrainIterator.toSeq)
```

## Gaussian Naive Bayes Classifier

Finally the classifier! We're building a Naive Bayes classifier with sample data having continuous pixel values.
Some can argue that actual pixel values are discrete - integers from 0 to 255 - but for the sake of problem we can take this as being continuous (they're not boolean values). So each pixel values assume Gaussian distribution.

Naive Bayes model takes an important **"naive"** assumption - conditional independence:

- each feature `x_i` is conditionally independent of all other `x_j`, if `i != j`

![](https://d2mxuefqeaa7sj.cloudfront.net/s_CC0EF2CDF71EE43DD6DFE4429654091813D72BFC55BC1C0440CEADAC21A906B2_1525762780409_file.jpeg)

This assumption, if you think about it, is actually false for MNIST dataset. 
Are pixels right above or right below a certain pixel truely independent?
Nevertheless, it is a 'safe enough' assumption. 
We can determine whether the digit is '0' given the test pixel values by whether certain pixel position were 'generally dark' for all other examples of '0' we're trained with.

The Naive Bayes classifier is finding the most probable 'class' given the dataset.
In other words, we're finding the class that makes the conditional probability `P(class | dataset)` the largest.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_CC0EF2CDF71EE43DD6DFE4429654091813D72BFC55BC1C0440CEADAC21A906B2_1525763071172_file.jpeg)

The conditional independence assumption simplifies the likelihood part.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_CC0EF2CDF71EE43DD6DFE4429654091813D72BFC55BC1C0440CEADAC21A906B2_1525763092396_file.jpeg)

### GaussianNaiveBayes trait

All we need for using the Naive Bayes classifier is prior and likelihood estimates.
For the sake of simple calculations, we use log-likelihood and log-prior instead. 
This will let us compute the sum of log-likelihoods instead of product of likelihoods.

```scala
trait GaussianNaiveBayes {
  def data: DataSet[Int, Seq[Float]]

  // calculate P(X | Y)
  def logLikelihood: Map[Int, Seq[Float => Option[Double]]]

  // calculate P(Y)
  def logPrior: Map[Int, Float]

  // calculate P(Y | X)
  def predict(x: Seq[Float]): Int
}
```


Using `MnistDataSet` data, we can now create a class `MnistGaussianNaiveBayesClassifier`.

```scala
case class MnistGaussianNaiveBayesClassifier(data: MnistDataSet) 
  extends GaussianNaiveBayes
```

### Parameter estimates

**log prior estimate**. Prior is the probability of each class' occurrence.
Log prior is just logs of them. MNIST should have prior estimates of roughly 1/10 for all classes.

```scala
lazy val logPrior: Map[Int, Float] = {
  lazy val totalSize = data.size
  data.groupByClass.mapValues(x => math.log(x.size.toFloat / totalSize).toFloat)
}
```


**log likelihood estimate**. 
For each attribute of input sample, the log likelihood is the gaussian distribution having the sample mean and variance. 
To compute this, we use `sampleMeanVariances` function of `MnistDataSet`, which uses `transpose` to group samples by 784 different attributes.
This was actually the nice part of functional programming, for inherently supporting [`numpy.transpose`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.transpose.html) from python world I'm used to.

`logLikelihood` maps each attribute to a function that takes an attribute value `Float` and outputs a `Double`, the likelihood. 
Mapping to a [curried](https://docs.scala-lang.org/tour/currying.html) function `gaussianProb(mean: Float, variance: Float)(x: Float): Double` makes it easier to compute the probability of a value that will be later on given (in this case, `x` - the test example's attribute value) while locking other parameters we can compute beforehand (in this case, `mean` and `variance` of each attribute).
This reduces the need to _remember_ the mean and variances of all attributes, as would other imperative languages do.
By currying the parameter `x` after `mean` and `variance`, a Gaussian distribution is defined, only waiting for an input value to compute the probability.

```scala
val epsilon = 0.1f
lazy val logLikelihood: Map[Int, Seq[Float => Option[Double]]] =
  // epsilon is added as a prior (hence making this an MAP estimate)
  data.sampleMeanVariances.mapValues(_.map({ case (mean, variance) =>
    gaussianProb(mean, variance + epsilon) _
  }))
```


**MAP and Gaussian prior**.
If a certain pixel contains a value 0 for all training samples, then its sample distribution will have both mean and variance 0.
The the probability of this pixel for a test sample will be 0, and this is a problem - the entire likelihood term will become 0 as it computes the product of all attributes' probabilities.
Thus we need to compute the [maximum a-posterior estimate (MAP)](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) instead of MLE. 
Setting the prior for each attribute as a normal distribution with known variance (`0.1f` in my case), the [posterior distribution](https://en.wikipedia.org/wiki/Conjugate_prior) will have variance of `sample variance + prior variance`.
This is why the `gaussianProb` accepts `variance + epsilon` as its variance parameter.

### Prediction

Now we know our priors and likelihoods, we are ready to predict which digit a handwritten number is.
This is as easy as computing the argmax of conditional probability of the mathematical equation written above.

The argmax function returns the key that has the maximum value.

```scala
def argmax[A](map: Map[A, Double]): A =
  map.keysIterator.reduceLeft((x, y) => if (map(x) > map(y)) x else y)
```

And the final predict function will be:

```scala
override def predict(x: Seq[Float]): Int = argmax(logPrior.map({
  case (cls, pri) =>
    val like = logLikelihood(cls).zip(x).flatMap { case (f, xVal) => f(xVal) }
    val posterior = like.sum + pri
    cls -> posterior  // map the class with posterior probability
}))
```


## Testing the classifier

With the following script, we can test the classifier with the first 10 test samples.

```scala
object Main {
  def main(args: Array[String]): Unit = {
    val mnist: MnistDataSet = Mnist.minstDataSet
    val mnistTest: MnistDataSet = Mnist.mnistTestDataSet
    val testData = mnistTest.data
    val classifier = MnistGaussianNaiveBayesClassifier(mnist)
    println("classifier loaded")
    println("Testing with 10 examples: ")
    val predictTestSet = testData.take(10).map(samp => (samp.label, classifier.predict(samp.sample)))
    println("Results: (correct_label, predicted_label)")
    println(predictTestSet.toVector)
  }
}
```

And the results show:

```
Results: (correct_label, predicted_label)
Vector((7,7), (2,2), (1,1), (0,0), (4,9), (1,1), (4,8), (9,9), (5,4), (9,9))
```

It predicts more than half of test samples correctly (which is good) while predicting some wrong.
Mispredicting a 4 with 5 seems especially odd.
However, with such a naive assumption of conditional independence, it is happy to see the algorithm actually works, although the accuracy is _horrible_ compared to state-of-the-art deep learning models giving error as low as [0.21%](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html).


## Conclusion

The test script runs *VERY SLOW*, and I will try to improve this. 
As a fresh-out-of-imperative-world, the performance optimization in functional programming language is still hard to grasp.

However, playing with functional programming for machine learning algorithms was an exciting challenge.
As easy as Naive Bayes classifier is, it already had plenty of rooms for functional programming ideas to practice with.
I look forward to write some iterative algorithms such as [EM](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) or [PCM](https://en.wikipedia.org/wiki/Principal_component_analysis) - which will be much more fun. 

Complete code used in this post is uploaded at [this github repository](https://github.com/dansuh17/scala-naive-bayes).

Until next time!
