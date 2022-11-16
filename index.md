# AdamW Analysis

## Setup
### Example of an env
We use a Voronoi partitioning of 2D space. We choose 2D space because it's easiest to visualize. We choose Voronoi partitions because it's very hard for a deep neural network to learn. 

![](vis_output/env.png)

<div style="page-break-after: always;"></div>


## Modeling
We start by creating a model that is able to learn in this environment and uses tools from state-of-the-art models today, namely ReLU, softmax, unbiased linear layers, and layernorms.

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

<div style="page-break-after: always;"></div>


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

<div style="page-break-after: always;"></div>


### ...And learning that values closer to 1 are better

We start slowly building towards our parameterized layernorm by first parameterizing the offset or bias. 

Notice that only parameterizing an offset without a scaling weight is poorly conditioned -- the eval loss is very noisy and there is a lot of variance in the classifications over steps. This is due to the nature of softmax -- values that are closer together more evenly distribute softmax weight which decreases expressivity. 

$$
\begin{align*}
& C_\theta = \text{Linear layer (} |\mathcal{C}||C| \text{ parameters)} \\
& \text{offset}_\theta = \text{Offset (1 parameter)} \\
& f_\theta = \text{softmax}\left(-\left|\frac{X^T\mathcal{C}}{\|\mathcal{C}\|_2^2} - \text{offset}_\theta\right|\right)C_\theta
\end{align*}
$$

![](results/LearnOffsetAdamWExperiment/benchmark.gif)

<div style="page-break-after: always;"></div>


### ...And learning that larger scales leads to better softmax outputs

$$
\begin{align*}
& C_\theta = \text{Linear layer (} |\mathcal{C}||C| \text{ parameters)} \\
& \text{offset}_\theta = \text{Offset (1 parameter)} \\
& \text{scale}_\theta = \text{Softmax conditioning scale (1 parameter)} \\
& f_\theta = \text{softmax}\left(-\text{scale}_\theta \left|\frac{X^T\mathcal{C}}{\|\mathcal{C}\|_2^2} - \text{offset}_\theta\right|\right)C_\theta
\end{align*}
$$

![](results/OffsetScaleAdamWExperiment/benchmark.gif)

<div style="page-break-after: always;"></div>


### ...And learning to use ReLU instead of absolute value

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

<div style="page-break-after: always;"></div>


### ...And learning the centers from scratch

$$
\begin{align*}
& C_\theta = \text{Linear layer (} |\mathcal{C}||C| \text{ parameters)} \\
& \text{offset}_\theta = \text{Offset (1 parameter)} \\
& \text{scale}_\theta = \text{Softmax conditioning scale (1 parameter)} \\
& \text{pm}_\theta = \text{Plus-minus parameter for learning absolute value (2 parameters)} \\
& \mathcal{C}_\theta = \text{Learned Voronoi centers (2048} D \text{ parameters)}\\
& f_\theta = \text{softmax}\left(-\text{scale}_\theta * \text{pm}_\theta * \text{ReLU}\left(\frac{X^T\mathcal{C}_\theta}{\|\mathcal{C}_\theta\|_2^2} - \text{offset}_\theta\right)\right)C_\theta
\end{align*}
$$

![](results/CentersAdamWExperiment/benchmark.gif)

<div style="page-break-after: always;"></div>


## Optimization

### AdamW L2 weight decay (default)

![](results/CustomAdamWExperiment/benchmark_scalars.gif)
![](results/CustomAdamWExperiment/benchmark_logits.gif)
![](results/CustomAdamWExperiment/benchmark_Centers.gif)

<div style="page-break-after: always;"></div>


### AdamW L1 weight decay

![](results/AdamWL1Experiment/benchmark_scalars.gif)
![](results/AdamWL1Experiment/benchmark_logits.gif)
![](results/AdamWL1Experiment/benchmark_Centers.gif)

<div style="page-break-after: always;"></div>


### AdamW L1 and L2 proportional weight decay

![](results/AdamWL1L2Experiment/benchmark_scalars.gif)
![](results/AdamWL1L2Experiment/benchmark_logits.gif)
![](results/AdamWL1L2Experiment/benchmark_Centers.gif)

<div style="page-break-after: always;"></div>


Next steps:
L1 reg promotes stability. L2 reg promotes learning through instability. Is there a way to reconcile them both?
