<head>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            TeX: {
                equationNumbers: {
                    autoNumber: "AMS"
                }
            },
            tex2jax: {
                inlineMath: [ ['$', '$'], ["\\(", "\\)"] ],
                displayMath: [ ['$$', '$$'], ["\\[", "\\]"] ],
                processEscapes: true,
            }
        });
        MathJax.Hub.Register.MessageHook("Math Processing Error",function (message) {
            alert("Math Processing Error: "+message[1]);
        });
        MathJax.Hub.Register.MessageHook("TeX Jax - parse error",function (message) {
            alert("Math Processing Error: "+message[1]);
        });
    </script>
    <script type="text/javascript" async
        src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
    </script>

</head>


## Setup
### Example of an env
We use a Voronoi partitioning of 2D space. We choose 2D space because it's easiest to visualize. We choose Voronoi partitions because it's very hard for a deep neural network to learn. 

![](vis_output/env.png)

<div style="page-break-after: always;"></div>


## Modeling
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
$$
\begin{align*}
& f_\theta = \arg \min_C \left|\frac{X^T\mathcal{C}}{\|\mathcal{C}\|_2^2} - 1\right|
\end{align*}
$$

![](results/ExactAdamWExperiment/benchmark.gif)

<div style="page-break-after: always;"></div>


### Learning which centers correspond to which class
$$
\begin{align*}
& C_\theta = \text{Linear layer (} |\mathcal{C}||C| \text{ parameters)} \\
& f_\theta = \text{softmax}\left(-\left|\frac{X^T\mathcal{C}}{\|\mathcal{C}\|_2^2} - 1\right|\right)C_\theta
\end{align*}
$$

![](results/CenterLabelsAdamWExperiment/benchmark.gif)

<div style="page-break-after: always;"></div>


### ...And learning that values closer to 1 are better
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

![](results/CustomAdamWExperiment/benchmark_vectors.gif)
![](results/CustomAdamWExperiment/benchmark_scalars.gif)

<div style="page-break-after: always;"></div>


### AdamW L1 weight decay

![](results/AdamWL1Experiment/benchmark_vectors.gif)
![](results/AdamWL1Experiment/benchmark_scalars.gif)

<div style="page-break-after: always;"></div>


### AdamW L1 and L2 proportional weight decay

![](results/AdamWL1L2Experiment/benchmark_vectors.gif)
![](results/AdamWL1L2Experiment/benchmark_scalars.gif)

<div style="page-break-after: always;"></div>
