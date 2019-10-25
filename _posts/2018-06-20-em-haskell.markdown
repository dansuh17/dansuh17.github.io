---
layout: post
title: "Machine Learning with Haskell - EM Algorithm and State Monads"
date:   2018-06-20 10:18:10 +0900
categories: jekyll update
---

Several months into teaching myself Haskell, I do now genuinely think Haskell is HOT. 
As an all-time Python lover, at first, I found Haskell codes mostly incomprehensible. 
However, I finally _kind of_ grasp the dark magic of monads.

<!--more-->

![](/assets/img/monad_dont_know.jpeg)

This article is about my little experiment on utilizing Haskell's monads in machine learning algorithm.

Most [intractable](https://en.wikipedia.org/wiki/Computational_complexity_theory#Intractability) algorithms in machine learning accompany iterative computations, optimizing the performance by updating parameters step-by-step.
Naturally, iteration and parameter updates connect to the concept of 'state's, or storage of variables.
The words __update__ and __variable__ do not sound appropriate for a functional language that constantly emphasizes "purity" and traumatically hates side-effects.
However, there is a _Haskellic_ way of dealing with states, using the `State` Monad.
In this article, I will try to explain my own implementation of one of the most popular iterative algorithms in machine learning, [EM (Expectation Maximization) algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm), making use of Haskell's `State` Monad.

## EM Algorithm and Coin Flip Example

*EM algorithm is an algorithm that tries to find the best parameter of a model when some of the data are missing or unobserved.*
It would be difficult to fully derive the EM algorithm in this article, but here's a brief explanation of a simple problem setting to implement (for a full derivation, refer to this nice [Stanford University's CS229 material](http://cs229.stanford.edu/notes/cs229-notes8.pdf).

Let's follow the popular coin flip example. Suppose the villain "Two-Face" now has two coins, `COIN_A` and `COIN_B`, to choose. These coins are inside a pouch and "Two-Face" randomly picks a coin inside the pouch without looking. We can say that the probability of each coin is 0.5. 

```
P(COIN_A) == P(COIN_B) == 0.5
```

![](https://www.dccomics.com/sites/default/files/GalleryComics_1920x1080_20171101__TWO-FACE_CELEBRATION_75_59d29736947264.79268719.jpg)

Each coin are biased, meaning that the coins do not give heads and tails with equal probability. If a coin's bias is 0.8, then out of 100 flips the coin is likely to give around 80 heads. 
"Two-Face" is being gentle and throws 10 flips once he picks a coin. 
If more than half of the throws give heads, the victim is doomed, otherwise spared. 
Now suppose that "Two-Face" _doesn't know_ how biased each coins are.

```
P(BIAS_A) = ?
P(BIAS_B) = ?
```

"Two-Face" has done this 'act of judgement' against 10 targets. The results are shown in this table;

| Coin (A or B)  | Flips               | Result (No. Heads) |
|----------------|---------------------|--------------------|
| ?              | H T T T T T T T T H | SPARED (2)         |
| ?              | H H H H T H H H H H | KILLED (9)         |
| ?              | T T H H H H T H H H | KILLED (7)         |
| ?              | H H H H H H T H T H | KILLED (8)         |
| ?              | H H H T H H H H H H | KILLED (9)         |
| ?              | H H T H H H T H H H | KILLED (8)         |
| ?              | T T T T T H T H T H | SPARED (3)         |
| ?              | T T H H H T T T T T | SPARED (3)         |
| ?              | H H H T H H H H T T | KILLED (7)         |
| ?              | H T T T T T T T H T | SPARED (2)         |

Clearly, the the coins are biased. Our job is to find estimates of coin biases, provided the train data - flip results for 10 targets.

## HMatrix - the numpy of Haskell

We now need to find a way to represent the numeric values needed during the calculations.
A really nice package for this purpose is [hmatrix](http://hackage.haskell.org/package/hmatrix).
Just like [numpy](http://www.numpy.org/) from Python world, `hmatrix` provides nice abstractions for representing vectors and matrices, as well as functions for performing linear algebra operations.

We can express the probability of coins as a `Vector` of real values (`R`). The `vector` function transforms a list of `Double`s into type `Vector R`.

```haskell
import Numeric.LinearAlgebra

probCoin :: Vector R  -- R is an alias for Double
probCoin = vector [0.5, 0.5]  -- P(COIN_A) == P(COIN_B) == 0.5
```

Similarly, the coin-flip results can also be expressed as a `Vector` containing the number of heads in 10 throws.

```haskell
headObserved :: Vector R  -- converted to R for convenience
headObserved = vector [2, 9, 7, 8, 9, 8, 3, 3, 7, 2]  -- number of heads
```

## Keeping the state with `State` Monad

Before proceeding with actual implementation of EM, we need to understand how `State` Monad will make sense for EM.
Haskell's [`State`](https://wiki.haskell.org/State_Monad) Monad allows us to pass around the state as we iterate.
As a sidenote, `State` Monad's name is somehow misleading since it doesn't actually contain any state values, but wraps a __state processing function__, which can be accessed with `runState`.
The state processing function gets the previous state as input and outputs both the (possibly) modified state and the output value as tuple.

```haskell
-- State [state type] [output type]
newtype State s a = State { runState :: s -> (a, s) }
```

EM is an itrative algorithm repeating the E-step and then the M-step until log-likelihood converges.

```python
# pseudocode
while not converged:
    model.e_step()
    model.m_step()
```
In our problem setting, the only parameters that gets updated are the estimation of coin biases (which "Two-Face" dosn't know!).
This is the only "state" we need to keep in our model.

```haskell
type Params = Vector R  -- coin biases
```

Then our type for states for `State` Monads in our models will be `Params`.

```haskell
State Params ?  -- State [state type] [output type]
```

The type for output values will be determined as we proceed with our implementation of EM step.
We have just went through the fist step of designing a model that can contain "state" using `State` monad!

## Parameter initialization

We don't know anything about coin biases (of course, we are doing all this mess in order to estimate the coin biases!).
However, in order to run the EM algorithm, we need to initialize the biases (parameters) with arbitrary probability values, say, 0.4 and 0.6. 

This is like saying: 

> Just imagine for now that the biases are 0.4 and 0.6. How probable are each trials for each coin?

This value is temporary and will be updated in the __M-Step__ described later on.

```haskell
-- initial coin bias
initParam :: Params
initParam vector [0.4, 0.6]
```

## E-Step

The main purpose E-Step (expecation step) is to fill in the blanks from missing (unobserved) data, using the temporary parameters (coin biases).
We must calculate the probabilities for the coins used for each trials (10 throws), given the bias and trials.
It is easy to derive the formula for this probability using [Bayes Rule](https://en.wikipedia.org/wiki/Bayes%27_theorem).

For example, calculating the probability of `COIN_A` and `COIN_B` for trial 1 (which gave 2 heads) will be:

```python
# pseudocode
EXP(COIN_A, TRIAL_1) 
    = P(COIN_A | TRIAL_1, COIN_BIAS) 
    = P(TRIAL_1 | COIN_A, COIN_BIAS) * P(TRIAL_1, COIN_BIAS) / sum_over_coin(P(TRIAL_1 | COIN_X, COIN_BIAS))
EXP(COIN_B, TRIAL_1) 
    = P(COIN_A | TRIAL_1, COIN_BIAS) 
    = P(TRIAL_1 | COIN_B, COIN_BIAS) * P(TRIAL_1, COIN_BIAS) / sum_over_coin(P(TRIAL_1 | COIN_X, COIN_BIAS))
```

Let's break down the terms in the expression above.

### the likelihood: `P(TRIAL_1 | COIN_A, COIN_BIAS)`

The expression `P(COIN_A | TRIAL_1, COIN_BIAS)` can be evaluated with binomial probability; 
the probability of getting `TIRAL_1` heads out of 10 throws of a coin (`COIN_A`) that has bias of `COIN_BIAS_A`.

```python
# pseudocode
# 1 - COIN_BIAS_A = probability of giving tails
# 10 - TRIAL_1 = number of tails in trial 1
P(COIN_A | TRIAL_1, COIN_BIAS) = Binom(TRIAL_1, 10, COIN_BIAS_A) = (COIN_BIAS_A ^ TRIAL_1) * ((1 - COIN_BIAS_A) ^ (10 - TRIAL_1))
```

The pseudocode above can be made a function in Haskell easily.

```haskell
binomProb :: Matrix R -> Matrix R -> Matrix R -> Matrix R
binomProb heads tails bias = (bias ** heads) * ((1 - bias) ** tails)
```

What's happening here: `binomProb` is getting inputs of `Matrix` types. 
`Matrix` simply represents the collection of training samples (i.e. number of heads).
Mathematical operators are applied element-wise, so `heads`, `tails` and `bias` all have same dimensions.
This way, we can simply multiply the matrices once instead of iterating through all training examples.

The `heads` matrix has size `(num_coins, num_samples)`, which is `(2, 10)` since we have two coins and ten trials.
`bias` matrix contains the bias of coins, replicated by the number of samples.
The output value will surely have size `(num_coins, num_samples)`, where each row represents the likelihood of each samples for a coin.
More of this will be clearer after we get to the prior.

### the prior: `P(TRIAL_1, COIN_BIAS)`

The prior is simple: the probability of picking the coin from pouch = `probCoin`.

### joint probability: `P(TRIAL_1 | COIN_A, COIN_BIAS) * P(TRIAL_1, COIN_BIAS)`

Multiplying the prior and the likelihood gives us the joint probability.

```haskell
-- probability that an event will happen = P(coin) * P(num_heads | coin)
jointProb :: Vector R -> Vector R -> Vector R -> Matrix R
jointProb numheads headBias coinprobs = coinprobMat * binomProb headMat tailMat headBiasMat
  where
    numCoins = size headBias  -- 2
    numSamples = size numheads  -- 10
    headMat :: Matrix R = fromRows $ replicate numCoins numheads  -- size (numCoins, numSamples)
    tailMat = 10 - headMat
    headBiasMat :: Matrix R = fromColumns $ replicate numSamples headBias
    coinprobMat :: Matrix R = fromColumns $ replicate numSamples coinprobs  -- size (numCoins, numSamples)
```

The `jointProb` defined above simply multiplies likelihood and prior element-wise, after formatting the input vectors appropriately into `Matrix`es in order to make computation easier.
The output matrix of `jointProb` will now contain the joint probabilities `P(COIN_i, TRIAL_j)` at (i, j).

### the denominator: `sum_over_coin(P(TRIAL_1 | COIN_X, COIN_BIAS))`

The denominator for the probability expression is there for normalization, in order to make the sum of `P(TRIAL_1 | COIN_X, COIN_BIAS)` over coins into 1.0.
We already have the values we need to sum over.
The direction we are summing over is the column-direction, so we create a function that does this.

```haskell
-- calculate the sum over colums
-- this is like: map sum [column_vectors]
columnSum :: Matrix R -> Matrix R
columnSum x = (1><dim) (repeat 1) <> x  -- matmul with size (1 x colsize) = sum over columns
  where
    dim :: Int = fst $ size x  -- dimension of axis 0 (size of column)

-- normalize a matrix across column (axis 1)
columnNormalize :: Matrix R -> Matrix R
columnNormalize x = x / colSum
  where
    colSum = columnSum x
```

The infix operator (><) just means to create a type `Matrix` having specific size. `(1><dim) (repeat 1)` is thus creating a `Matrix` of size (1, dim) filled with 1's.
Another infix operatior (<>) is matrix multiplication.

### probability of coins per sample.

We have prepared everything we need in order to evaluate the expected values of coins for each sample (trial).

```haskell
-- E-step => probability of coins
-- each row (index axis 0) => examples
-- each columns (index axis 1) => coin's expected values (per example)
-- resulting size : (num_coins, num_examples)
coinProbEst :: Vector R -> Vector R -> Vector R -> Matrix R
coinProbEst heads biases coinprobs = columnNormalize jointp
  where
    jointp = eventProb heads biases coinprobs  -- joint probability
```

Calculating `coinProbEst` for our coin flip example, and then transposing it (for visual purpose), we get:

```
(10><2)
 [   0.9192938209331653, 8.070617906683486e-2
 , 3.755317588381989e-2,   0.9624468241161802
 ,  0.16494845360824748,   0.8350515463917526
 , 8.070617906683486e-2,   0.9192938209331653
 , 3.755317588381989e-2,   0.9624468241161802
 , 8.070617906683486e-2,   0.9192938209331653
 ,   0.8350515463917526,  0.16494845360824748
 ,   0.8350515463917526,  0.16494845360824748
 ,  0.16494845360824748,   0.8350515463917526
 ,   0.9192938209331653, 8.070617906683486e-2 ]
```

where each row are the samples and the values in columns each denote the probability of coins for that sample.
For example, in the first row, the probability that `COIN_A` was coin that gave the result is 92% while the probability that `COIN_B` was the coin is just 8%.
Since trial 1 gave only 2 heads, `COIN_A` having initial bias of 0.4 is __MUCH__ more probable than `COIN_B`, having bias of 0.6.

## M-Step

Now that we have the probabilities of coin labels for each trials, we need the compute the parameters (= coin biases) that __maximizes__ the data likelihood (hence the M-step).
The bias values that maximizes the data probability is expressed as:

```python
# pseudocode
UPDATED_BIAS_A = TOTAL_HEADS_FOR_COIN_A / TOTAL_THROWS_FOR_COIN_A
```

This is intuitive. If a coin gave 8 heads out of 10 throws, we would normally think that the coin has a bias of about 0.8.
However, we only have the probabilities of coins for each trials.
Instead of _counting_ the number of heads, we need to evaluate the _weighted sum_ of heads, where the weights are the probability of coin that gave those heads.
For example, for the first trial that gave 2 heads, 92% is attributed to `COIN_A` while other 8% is attributed to `COIN_B`.
So the number of heads that `COIN_A` gave for trial 1 is `0.92 * 2 = 1.84`.

Weighted sum can be directly interpreted as taking the dot product of probability vector and heads vector.
Using this interpretation, we can go ahead and update the parameters.

```haskell
-- calculate updated parameters
calcUpdatedParams :: Vector R -> Matrix R -> Vector R
calcUpdatedParams obsvd exps = fromList $
  map (\weightVec ->  -- expeced values for coin
          weightVec <.> obsvd / weightVec <.> totalThrows)  -- weighted sum = dot product
      sampleExpected
  where
    -- Expected values for samples per coin. If there are two coins, length sampleExpected == 2
    sampleExpected :: [Vector R] = toRows exps  -- becomes a list of : [expected values of all samples for coin]
    totalThrows :: Vector R = vector $ replicate numSamps totalThrow
    numSamps :: Int = size $ head sampleExpected
```

The dot product `weightVec <.> obsvd` replaces the `TOTAL_HEADS_FOR_COIN_A` in the pseudocode above. 
In the same manner, `weightVec <.> totalThrows` replaces the `TOTAL_THROWS_FOR_COIN_A`.

Calculating the updated parameters once with our problem settings will yield new bias values: 0.318 for `COIN_A`, and 0.760 for `COIN_B`.
The results are, again, intuitive.
The trials that spared the targets only gave 2 or 3 heads, and initial bias of 0.4 seems too high. After the update, it has been lowered to 0.318.
Similarly, the number of heads that killed the targets gave around 7 to 9 heads, and initial bias of 0.6 is definitely too low; after the update, it has been incremented to 0.76.

## Iteration and State Monad

With `calcUpdatedParams`, we can now update the coin biases - but how do we iterate the update process?
Moreover, where do we store the intermediate values?
We can do some recursions, providing the iteration number, stopping when iteration number reaches zero, appending intermediate parameter values as we recurse, and passing _THEM_ around.
This is dirty. What a mess!

We can use `State` Monad introduced above to answer both questions.

### redefining functions using State Monad

Remind that the parameters of our model are coin bias values.

__In the E-step__, instead of directly returning the coin probability matrix, `coinProbEst` can be modified to return a `State`.
This state is a wrapper of state-processing function that takes the previous parameters (coin biases) and outputs both then next parameters and coin probability matrix.

```haskell
coinProbEst :: Vector R -> Vector R -> State Params (Matrix R)
coinProbEst heads coinProbs = state $ \params ->
  (columnNormalize (eventProb heads params coinProbs), params)  -- doesn't modify the state
```

Note that during the E-step doesn't update the coin bias values, so the input state is passed on without being modified.
`state` function is used to create a `State` Monad by providing the state processing function (that should have type `s -> (a, s)`, which is `Params -> (Matrix R, Params)` in this case).

__In the M-step__, `calcUpdatedParams` can be used also to create a `State`. 
State-processing function in this `State` monad will contain the updated parameters.

```haskell
updateParams :: Vector R -> Matrix R -> State Params ()
updateParams heads probcoin = state $ const ((), calcUpdatedParams heads probcoin)  -- pass on the updated state
```

__Parameter initialization__ can also be expressed using `State` using `put` function, which contains a state-processing function that replaces the state being passed around.

```haskell
-- input = initial coin bias values
initEm :: Vector R -> State Params ()
initEm = put

-- note the definition of put
put newState = state $ \_ -> ((), newState)
```

### combination and iteration

One EM step contains one E-Step and one M-Step. 
We can combine these two steps easily with `bind`, now that we have monads.

```haskell
emStep :: Vector R -> Vector R -> State Params ()
emStep heads coinProbs = coinProbEst heads coinProbs >>= updateParams heads
```

The `(>>=)` operator, or monadic `bind`, accepts a `State` monad first and a function. 
The output value of `State` monad's processing function (in this case, the output of `coinProbEst`'s `State` is the coin probabilities) is then applied to the function, 
creating the next `State` monad (in this case the output `State` of `updateParams`).

We can make use of the fact that two other values required for iteration - observed heads and probability of coins being picked from the pouch - are __constant__.
The two constants are `headObserved` and `probCoin`, respectively.
This leads to a simplified wrapper `State` that runs one EM-step.

```haskell
-- one stepper state for EM
-- after the step, return the current state
emStepper :: State Params Params
emStepper = emStep headObserved probCoin >> get
```

`get` function used here simply returns the current state as output.

```haskell
get = state $ \s -> (s, s)
```

Now, performing the iteration is as simple as calling `replicateM` over `emStepper` monad!

```haskell
-- iterate EM algorithm multiple times and collect intermediate params
emIter :: Int -> State Params [Params]
emIter numIter = replicateM numIter emStepper
```

This is how stateful iterations are made in Haskell. No for-loops, no while-loops.

## Results & Conclusion

Checking out the intermediate values after 10 iterations, after initializing the coin biases with `[0.4, 0.6]` is expressed as:

```haskell
-- intermediate parameters
intermParams :: [Vector R] = evalState (initEm (vector [0.1, 0.3]) >> emIter 10) $ vector []
```

`evalState` takes a `State` and an initial state value (of type `Params`), and returns the output value after all state-processing is finished.
The empty vector `vector []` provided here is meaningless because it is overwritten by `initEm`.

The results of 10 iterations are as follows (with some pretty-printing);

```
Step : 1  Params: [0.3181271315275837,0.7601145899025433]
Step : 2  Params: [0.26299184390385155,0.799827186851992]
Step : 3  Params: [0.2550167719144156,0.8001081427495718]
Step : 4  Params: [0.2541708988752831,0.7999958870233562]
Step : 5  Params: [0.25408453342034,0.7999816586068763]
Step : 6  Params: [0.25407572598399125,0.7999801541376964]
Step : 7  Params: [0.25407482733626807,0.7999799996411089]
Step : 8  Params: [0.2540747356291876,0.7999799838568136]
Step : 9  Params: [0.2540747262701262,0.7999799822456467]
Step : 10  Params: [0.25407472531499103,0.7999799820812141]
```

The bias for `COIN_A` converges to 0.25, while bias for `COIN_B` converges to 0.8.
Let's hope we made "Two-Face" happy with our results!
The results are actually consistent with the fact that I have actually generated the 10 trial data presented above from random binomial distribution having bias of 0.25 and 0.8 each.


---


This was my little experiment spawned from curiosity about how powerful Haskell's monads can be, even when implementing machine learning algorithms.
Machine learning without python was unimaginable at first, but the joy playing with monads and actually having generated this nice results with EM algorithm was thrilling and exciting than I have ever imagined 
(Look at the iterations using monads in `emIter`! A one-liner!?).
I have come to think that functional languages are even better-suited for machine learning than python or any other imperative languages are.
Such thoughts have been more boosted after reading this [article](http://colah.github.io/posts/2015-09-NN-Types-FP/) about neural networks and functional programming.

I will also have to admit, though, that learning Haskell and `State` Monads was painful. 
So much headaches, I haven't attempted to use even more awesome concepts like [GADTs](https://en.wikibooks.org/wiki/Haskell/GADT), [Dependent Types](https://www.schoolofhaskell.com/user/konn/prove-your-haskell-for-great-safety/dependent-types-in-haskell), and/or [DataKinds](https://www.schoolofhaskell.com/user/k_bx/playing-with-datakinds).
Even so, there might be some deeper _something_ between machine learning and functional programming, and I would be very glad to see how those two get along in the future.


- The full codes can be seen in this [repository](https://github.com/deNsuh/haskell-em), and the following is the full haskell code for this article:

```haskell
{-# LANGUAGE ScopedTypeVariables #-}
module EmCoinState where

import Control.Monad.Trans.State (State, put, state, get)
import Control.Monad (replicateM)
import Numeric.LinearAlgebra
    (Vector
    , R
    , Matrix
    , vector
    , fromList
    , toList
    , fromRows
    , (<.>)  -- dot product (vector)
    , (><)  -- matrix formation
    , (<>)  -- matmul
    , size
    , toRows
    , fromColumns
    )

-- total number of throws
totalThrow :: R
totalThrow = 10

-- initial parameters
-- initial bias of two coins giving heads
initParam :: Params  -- vector of real values - R is just an alias of Double
initParam = vector [0.4, 0.6]

-- observed data represented by the number of heads in 10 coin-flips
headObserved :: Vector R
headObserved = vector [2, 9, 7, 8, 9, 8, 3, 3, 7, 2]

-- test the validity of observed data - should be less than 10
testObserved :: Vector R -> Bool
testObserved = foldl (\acc hd -> ((hd <= 10) && acc)) True . toList

-- probability of coin being generated
probCoin :: Vector R
probCoin = vector [0.5, 0.5]

-- define type parameters
type Params = Vector R

-- binomial probability
binomProb :: Matrix R -> Matrix R -> Matrix R -> Matrix R
binomProb hs tails bs = (bs ** hs) * ((1 - bs) ** tails)

-- probability that an event will happen = P(coin) * P(num_heads | coin)
eventProb :: Vector R -> Vector R -> Vector R -> Matrix R
eventProb numheads headBias coinprobs = coinprobMat * binomProb headMat tailMat headBiasMat
  where
    numCoins = size headBias
    numSamples = size numheads
    headMat :: Matrix R = fromRows $ replicate numCoins numheads
    tailMat = 10 - headMat
    headBiasMat :: Matrix R = fromColumns $ replicate numSamples headBias
    coinprobMat :: Matrix R = fromColumns $ replicate numSamples coinprobs

-- calculate the sum over colums
-- this is like: map sum [column_vectors]
columnSum :: Matrix R -> Matrix R
columnSum x = (1><dim) (repeat 1) <> x  -- matmul with size (1 x colsize) = sum over columns
  where
    dim :: Int = fst $ size x  -- dimension of axis 0 (size of column)

-- normalize a matrix across column (axis 1)
columnNormalize :: Matrix R -> Matrix R
columnNormalize x = x / colSum
  where
    colSum = columnSum x

-- E-step == expected values of coins
-- each row (index axis 0) => examples
-- each columns (index axis 1) => coin's expected values (per example)

-- P(coin_i | observed, theta) = P(observed | coin_i, theta) * P(coin_i) / sum_over_k( P(observed | coin_k, theta) * P(coin_k) )
-- prod_over_x ( binom x_head (10 - x_head) coin_i_bias * 0.5 )
-- resulting size : (num_coins, num_examples)
coinProbEst :: Vector R -> Vector R -> State Params (Matrix R)
coinProbEst heads coinProbs = state $ \params ->
  (columnNormalize (eventProb heads params coinProbs), params)  -- doesn't modify the state

-- M-step = calculate updated theta
-- new_theta_coin = sum (weighted heads) / sum (weighted total_throws)
updateParams :: Vector R -> Matrix R -> State Params ()
updateParams heads coinExps = state $ const ((), calcUpdatedParams heads coinExps)

-- calculate the next parameters
calcUpdatedParams :: Vector R -> Matrix R -> Vector R
calcUpdatedParams obsvd exps = fromList $
  map (\weightVec ->  -- expeced values for coin
          weightVec <.> obsvd / weightVec <.> totalThrows)  -- weighted sum = dot product
      sampleExpected
  where
    -- Expected values for samples per coin. If there are two coins, length sampleExpected == 2
    sampleExpected :: [Vector R] = toRows exps  -- becomes a list of : [expected values of all samples for coin]
    totalThrows :: Vector R = vector $ replicate numSamps totalThrow
    numSamps :: Int = size $ head sampleExpected

-- initial coin bias values
initEm :: Vector R -> State Params ()
initEm = put

-- state contains the parameters.
-- TODO : make state processing function output a log-likelihood + current state
emStep :: Vector R -> Vector R -> State Params ()
emStep heads coinProbs = coinProbEst heads coinProbs >>= updateParams heads

-- one stepper state for EM
-- after the step, return the current state
emStepper :: State Params Params
emStepper = emStep headObserved probCoin >> get

-- iterate EM algorithm multiple times and collect intermediate params
emIter :: Int -> State Params [Params]
emIter numIter = replicateM numIter emStepper
```
