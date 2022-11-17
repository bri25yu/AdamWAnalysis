# AdamW Analysis

## Setup
### Example of an env
We use a Voronoi partitioning of 2D space. We choose 2D space because it's easiest to visualize. We choose Voronoi partitions because it's very hard for a deep neural network to learn so the results will be applicable to other environments. 

![](vis_output/env.png)


## Modeling
### Motivation
We start by creating a model that is able to learn in this environment and uses tools from state-of-the-art models today, namely ReLU, softmax, and unbiased linear layers.

There are certain structures present in SOTA models that are not present in the following models, namely layernorms. Layernorming the input would literally change the actual input since the particular magnitude of an input actually matters for this environment, so it introduces an unnegotiable bias. Layernorming in transformers does not increase the bias significantly as the magnitude of a vector is not critical. 

There are certain structures that will be present in the following models that are not present in SOTA models. These structures are used deliberately and sparingly to actually make learning in this environment tractable.


### Experiment setup
All experiment runs use:
- learning rate of 1e-2. This is very large compared to today's models which have a maximum learning rate of 1e-4 to 1e-3
- weight decay of 1e-2, resulting in decay equal to 1e-2 * 1e-3 = 1e-5 times the parameter weight
- 10k steps. This is very typical for today's models
- batch size of 32. This is very small compared to today's models which typically have a batch size of 256 to 1024, making this challenge tougher
- 10k steps * 32 points per step = 320k datapoints seen over the course of training
- Eval and test set sizes of 10k


### Math setup
All of the following equations use the following shared definitions.

$$
\begin{align*}
& N = \text{batch size} \\
& D = \text{dimension of environment} \\
& X = \text{inputs to classify of shape (N, D)} \\
& \mathcal{C} = \text{Voronoi centers} \\
& C = \text{classes} \\
& \theta = \text{model parameters} \\
& \mathcal{L}(\hat{y}, y) = \text{loss function. Crossentropy in this case} \\
& f_\theta = \text{classifier} \\
& J_\theta = \text{objective function} = \mathcal{L}(f_\theta(X), y)\\
\end{align*}
$$


### Exact algorithm

This is the algorithm with full information. The fact that it has 100% accuracy but nonzero loss is an artifact of crossentropy loss, where 0/1 predictions output a minimum of $\frac{1}{e+1}$ loss.

Specifically, the algorithm has access to:
- centers (and the fact that there are centers)
- classes and which centers correspond to which classes
- function to map an input to its corresponding center and class


$$
\begin{align*}
& f_\theta = \arg \min_C \left\lvert\frac{X^T\mathcal{C}}{\|\mathcal{C}\|_2^2} - 1\right\rvert
\end{align*}
$$

![](results/ExactAdamWExperiment/benchmark.gif)


### Learning which centers correspond to which class

Very easy classification task, given that the model knows a lot about the structure of the environment. 

The parameterized classification head is able to better condition to the crossentropy loss, allowing the loss to reach nearly 0. The majority class weight increases to positive infinity and the minority class weight decreases to negative infinity, which are probability 1 and 0 after softmax respectively. 

This model doesn't reach 100% accuracy, probably because there's very small Voronoi compartments that the model hasn't seen training data for yet.

$$
\begin{align*}
& C_\theta = \text{Linear layer (} |\mathcal{C}||C| \text{ parameters)} \\
& f_\theta = \text{softmax}\left(-\left\lvert \frac{X^T\mathcal{C}}{\|\mathcal{C}\|_2^2} - 1\right\rvert \right)C_\theta
\end{align*}
$$

![](results/CenterLabelsAdamWExperiment/benchmark.gif)

This is a good example of different classes of convergence for different optimizers. Obviously convergence depends on a lot of factors, but here's a good starting point: the best stochastic optimizers will converge to optimums even given non-convergent sequences of gradients. 

The prototypical scenario is when we have multiple steps of small gradient then one step of large gradient. The prototypical example is imbalanced classification, where minibatches of samples consist of the majority class and rarely contain the minority class which incur a large loss. The idea is that the optimizer is never able to really settle into either stationary point, either the local optimum of just the majority class or the global optimum of both classes. 

Let $\mathcal{H}$ be our optimization horizon, in this case $\mathcal{H} = 10000$.

1. Does the loss converge? $\limsup_{t \rightarrow \mathcal{H}} \mathcal{L}_t \stackrel{?}{=} \liminf_{t \rightarrow \mathcal{H}} \mathcal{L}_t$

2. Does there exist a stationary distribution over the parameters?
   - There exists no notion of absolute deterministic convergence with our stochastic optimization methods especially with regularization techniques such as weight decay.
   - Let $\mathcal{T}$ be the update operator. $\stackrel{?}{\exists} \pi^*_\theta \text{ s.t. } \pi^*_\theta = \pi^*_\theta \mathcal{T}$

3. Do the parameters converge to that stationary distribution? $\lim_{t \rightarrow \mathcal{H}}\pi^t_\theta = \pi^*_\theta$

4. How quickly do the parameters converge to the stationary distribution?

5. Finally we can start considering optimality

We revisit these ideas later throughout.


### ...And learning that values closer to 1 are better

We parameterize the offset/bias.

Notice that only parameterizing an offset without a scaling weight is poorly conditioned -- the eval loss is very noisy and there is a lot of variance in the classifications over steps. This is due to the nature of softmax -- values that are closer together more evenly distribute softmax weight which decreases expressivity. 

Another reflection is that the learning rate used was half the learning rate of other experiments, 5e-3 rather than 1e-2. Otherwise the model wouldn't actually fit to the data.

$$
\begin{align*}
& C_\theta = \text{Linear layer (} |\mathcal{C}||C| \text{ parameters)} \\
& \text{offset}_\theta = \text{Offset (1 parameter)} \\
& f_\theta = \text{softmax}\left(-\left|\frac{X^T\mathcal{C}}{\|\mathcal{C}\|_2^2} - \text{offset}_\theta\right|\right)C_\theta
\end{align*}
$$

![](results/LearnOffsetAdamWExperiment/benchmark.gif)


### ...And learning that larger scales leads to better softmax outputs

We introduce a scaling parameter for better softmax conditioning. The softmax logits and resulting softmax probabilities actually yield sizeable differences and the eval loss and predictions are much more stable. 

This scale parameter is also related to entropy in the sense that as we progress, we increase our confidence and reduce the entropy of our predictions. 

The scale and offset parameters are specific to this environment, where learning is intractable without them. The loss typically doesn't reach below 0.65 and the accuracy hovers just above MLE accuracy.

$$
\begin{align*}
& C_\theta = \text{Linear layer (} |\mathcal{C}||C| \text{ parameters)} \\
& \text{offset}_\theta = \text{Offset (1 parameter)} \\
& \text{scale}_\theta = \text{Softmax conditioning scale (1 parameter)} \\
& f_\theta = \text{softmax}\left(-\text{scale}_\theta \left|\frac{X^T\mathcal{C}}{\|\mathcal{C}\|_2^2} - \text{offset}_\theta\right|\right)C_\theta
\end{align*}
$$

![](results/OffsetScaleAdamWExperiment/benchmark.gif)


### ...And learning to use ReLU instead of absolute value

We parameterize our absolute value, forcing the model to learn that dot product values farther from 1 are worse. 

The two new multiplicative parameters and the scale both increase in magnitude over time, which causes more instability as reflected by the noisier loss curve. 

After this step, we have generalized our model so that it uses the same structures as SOTA models. 

$$
\begin{align*}
& C_\theta = \text{Linear layer (} |\mathcal{C}||C| \text{ parameters)} \\
& \text{offset}_\theta = \text{Offset (1 parameter)} \\
& \text{scale}_\theta = \text{Softmax conditioning scale (1 parameter)} \\
& \text{pm}_\theta = \text{Plus-minus parameter for learning absolute value (2 parameters)} \\
& f_\theta = \text{softmax}\left(-\text{scale}_\theta * \text{pm}_\theta * \text{ReLU}\left(\frac{X^T\mathcal{C}}{\|\mathcal{C}\|_2^2} - \text{offset}_\theta\right)\right)C_\theta
\end{align*}
$$

![](results/PlusMinusAdamWExperiment/benchmark.gif)


### ...And learning the centers from scratch
We parameterize the center locations. Every aspect of our model is parameterized now and the model starts with very little previous information.

Note that as soon as we overparameterize our model, model values that should converge no longer converge. More analysis on this in the next section.

$$
\begin{align*}
& H = \text{hidden size} \\
& C_\theta = \text{Linear layer (} |\mathcal{C}||C| \text{ parameters)} \\
& \text{offset}_\theta = \text{Offset (1 parameter)} \\
& \text{scale}_\theta = \text{Softmax conditioning scale (1 parameter)} \\
& \text{pm}_\theta = \text{Plus-minus parameter for learning absolute value (2 parameters)} \\
& \mathcal{C}_\theta = \text{Learned Voronoi centers (} HD \text{ parameters)}\\
& f_\theta = \text{softmax}\left(-\text{scale}_\theta * \text{pm}_\theta * \text{ReLU}\left(\frac{X^T\mathcal{C}_\theta}{\|\mathcal{C}_\theta\|_2^2} - \text{offset}_\theta\right)\right)C_\theta
\end{align*}
$$

![](results/CentersAdamWExperiment/benchmark.gif)

### [Work in progress] Generalizing parameter structures

This is run with learning rate of 2e-2 and a weight decay of 0. 

$$
\begin{align*}
& H = \text{hidden size} \\
& C_\theta = \text{Linear layer (} |\mathcal{C}||C| \text{ parameters)} \\
& \text{scale}_\theta = \text{Softmax conditioning scale (H parameters)} \\
& \mathcal{C}_\theta = \text{Learned Voronoi centers with bias (} H(D + 1) \text{ parameters)}\\
& f_\theta = \text{softmax}\left(-\text{scale}_\theta * \left|X^T\mathcal{C}_\theta\right|\right ) C_\theta
\end{align*}
$$

![](results/SwishAdamWExperiment/benchmark.gif)

## Optimization
All experiments in this section use the same parameters as the modeling section.


### AdamW L2 weight decay (default)

L2 weight decay introduces very large bias for large weights. This causes our weights to not converge, even while the loss more or less has. This also causes some catastrophic forgetting, where the model is able to find good explanations for the data, but after weight decay it forgets those explanations and the loss increases. 

$$
\theta_t \gets \theta_{t-1} - \gamma \lambda \theta_{t-1}
$$

![](results/CustomAdamWExperiment/benchmark_scalars.gif)
![](results/CustomAdamWExperiment/benchmark_logits.gif)
![](results/CustomAdamWExperiment/benchmark_Centers.gif)


### AdamW L1 weight decay
L1 weight decay introduces much less bias for large weights, which allows the model to learn more (hence the lower loss), and it allows our model parameters to actually converge. However, the stability in our model parameters actually causes instability in our loss because the model has more modeling capacity due to lower bias than in the L2 case. 

$$
\theta_t \gets \theta_{t-1} - \gamma \lambda \text{sign}(\theta_{t-1})
$$

![](results/AdamWL1Experiment/benchmark_scalars.gif)
![](results/AdamWL1Experiment/benchmark_logits.gif)
![](results/AdamWL1Experiment/benchmark_Centers.gif)


## Next steps
- Fully general parameterized model
- Compromising on bias introduced by L2 weight decay and instability introduced by L1 weight decay
    - L2 weight decay dropout?
    - L1 weight decay with EMA parameter statistics?
    - Gradient dependent weight decay?
      - Small gradients signal crystallization -> impose strong weight decay
      - Large gradients signal learning -> impose weak weight decay
