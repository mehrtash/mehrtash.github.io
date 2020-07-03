---
title:  "Confidence Calibration and Uncertainty Estimation for Deep Medical Image Segmentation"
mathjax: true
layout: post
type: blog
---

<!-- introduction -->
[Fully convolutional neural networks](https://arxiv.org/abs/1411.4038) (FCNs) have been successfully used 
in medical image analysis for semantic segmentation of normal organs and lesions.
The [U-Net](https://arxiv.org/abs/1505.04597), which is arguably the most popular architecture in medical image segmentation, 
has achieved top ranking results in several segmentation challenges.
Additionally, [Batch Normalization](https://arxiv.org/abs/1502.03167) (BN) and 
[Dice loss](https://arxiv.org/abs/1606.04797) are often used stabilize and accelerate training.
Models trained with BN tend to be [less calibrated](https://arxiv.org/abs/1706.04599).
Also, [empirical results on cardiac MRI segmentation](https://arxiv.org/abs/1809.10430) suggest that 
networks trained with Dice loss are often poorly calibrated.
[Miscalibration is a known issue in modern neural networks](https://arxiv.org/abs/1706.04599). 

Calibration is described as the ability of a decision-making system to provide an expectation of success (i.e. correct classification).
Using a frequentist interpretation of uncertainty, predictions (i.e. class probabilities) of a 
*well-calibrated* model should match the probability of success of those inferences in the long run.
For instance, if a well-calibrated brain tumor segmentation model classifies 100 pixels each with the 
probability of 0.7 as cancer,  we expect 70 of those pixels to be correctly classified as cancer. 
Miscalibration results in models that are unreliable and hard to interpret.
In some domains, for example medical applications, or automated driving, [overconfidence can be dangerous](https://arxiv.org/abs/1606.06565).


In our recent [IEEE TMI](https://ieeexplore.ieee.org/document/9130729) paper [“Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical 
Image Segmentation”](https://arxiv.org/abs/1911.13273), we study predictive uncertainty 
estimation in FCNs for medical image segmentation.
We propose model ensembling for calibration of FCNs trained with Dice loss.
We also present an entropy-based metric to predict segmentation quality of foreground structures at inference time,
which can be also used to detect out-of-distribution samples.
We conduct experiments across three medical image segmentation applications
of [brain](https://www.med.upenn.edu/sbia/brats2017/data.html), 
[heart](https://www.creatis.insa-lyon.fr/Challenge/acdc/), 
and [prostate](http://isgwww.cs.uni-magdeburg.de/cas/isbi2018/) to evaluate our contributions.

<figure class="figure">
  <img src="../assets/images/posts/2020-07-02-uncertainty-estimation/dice_vs_ce.png" class="figure-img img-fluid rounded" alt="A generic square placeholder image with rounded corners in a figure.">
  <figcaption class="figure-caption">A caption for the above image.</figcaption>
</figure>

#### Metrics for Calibration Quality Assessment
We use 
[Negative Log Likelihood (NLL)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html), 
[Brier score](https://en.wikipedia.org/wiki/Brier_score),  and 
[reliability diagrams](https://arxiv.org/abs/1706.04599) for evaluating calibration and uncertainty estimation.
We also use [Expected Calibration Error (ECE)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4410090/)
as a quantitative summary of the reliability diagrams.

#### Dice loss vs Cross Entropy (CE)
We start by comparing baselines trained with weighted cross-entropy (CE) loss 
with those that were trained with Dice loss in terms 
of segmentation quality and predictive uncertainty estimation.
We consistently observe that FCNs trained with Dice loss perform better segmentation compared to 
those trained with CE (in terms of Dice score and Hausdorff distance) but at the cost of worse calibration
(in terms of NLL and ECE%).

#### Calibration by Ensembling
We propose [ensembling](https://web.engr.oregonstate.edu/~tgd/publications/mcs-ensembles.pdf) for confidence calibration of poorly calibrated FCNs trained with Dice loss. 
Similar to the [Deep Ensembles](https://arxiv.org/abs/1612.01474) method, 
we train $$M$$ FCNs from scratch, with random initialization 
of the network parameters and random shuffling of the training dataset (using different random seeds 
for each of the baselines).
However, unlike the Deep Ensemble method, we do not use any form of adversarial training.
At inference time, we compute the probability of the ensemble as the average of the baseline probabilities.
We observe that ensembling significantly improves the calibration of models, specifically those that
were trained with Dice loss. For instance, for the prostate, the heart, and the brain tumor segmentation, 
using even five ensembles (M=5) of baselines trained with Dice loss can reduce the NLL by 
about 66%, 44%, and 62%, respectively.

<figure class="figure">
  <img src="../assets/images/posts/2020-07-02-uncertainty-estimation/n_models.png" class="figure-img img-fluid rounded" alt="A generic square placeholder image with rounded corners in a figure.">
  <figcaption class="figure-caption">A caption for the above image.</figcaption>
</figure>

#### Comparison with MC Dropout
[Monte Carlo dropout (MC dropout)](https://arxiv.org/abs/1506.02142) is widely used 
for uncertainty estimation of deep networks.
In this method dropout layers are applied at certain layers during the training time.
At test time, the probabilistic Gaussian process (uncertainty) is approximated 
by running the model several times with active dropout layers.
Here, we also compared ensembling with [MC dropout](https://arxiv.org/abs/1511.02680). 
We observe that in all cases ensembling outperforms MC dropout models in 
terms of calibration and segmentation quality.
Another benefit of ensembling over MC dropout is that it does not enforce the use of dropout layers.


#### Segment-level Predictive Uncertainty Estimation
it is often desirable to have a confidence metric that captures model uncertainty at the segment-level
of a test image volume.
Let's assume that we have a pelvic MRI volume and we use a deep model to segment the prostate gland.
If we have access to the ground truth, we can calculate Dice score of the prostate and 
describe the performance of the model for this specific example with a scalar metric.
But what if we don't have access to the ground truth.
Is there any way to come up with a scalar metric that is a good representative of the segmentation quality
of the predicted prostate gland?
In other words we seek a metric that has an acceptable correlation with Dice score and can be calculated 
at test-time without any knowledge of the ground truth.
Such a metric would be useful in clinical deployment of deep models.
Here, we propose an entropy-based metric as a segmnet-level uncertainty estimation tool.
Given the pixel-level class predictions $$\hat{y}_i$$ and their associated ground truth class 
$$y_i$$ for a predicted segment $$\hat{\mathcal{S}}_k = \{s \in (x_i, \hat{y}_i) | \hat{y}_i=k \}$$, 
we propose to use the average of pixel-wise entropy values over the predicted foreground:

$$\overline{\mathcal{H}(\hat{\mathcal{S}}_k)} = - \frac{1}{\left|\hat{\mathcal{S}}_k\right|} \sum_{i\in \hat{\mathcal{S}}_k} 
[p(\hat{y}_i=k| x_i,\theta)\cdot \ln{\left(p(\hat{y}_i=k|x_i,\theta)\right)} + \\
     \left(1- p(\hat{y}_i=k| x_i,\theta) \right) \cdot \ln{\left(1-p(\hat{y}_i=k|x_i,\theta)\right)}].$$
     
In calculating the average entropy of $$\hat{\mathcal{S}}_k$$, we assumed binary classification: 
the probability of belonging to class $$k$$, $$p(\hat{y}_i=k| x_i, \theta)$$ 
and the probability of belonging to other classes $$1 - p(\hat{y}_i=k| x_i, \theta)$$.

We calculate segment-level confidence for each of the foreground 
labels and plot the $$\overline{\mathcal{H(\hat{\mathcal{S})}}}$$ vs. Dice.
We observed a strong correlation between Dice coefficient and average of entropy over the predicted segment.
Hence, $$\overline{\mathcal{H(\hat{\mathcal{S})}}}$$ can be used as a useful metric for 
predicting the segmentation quality of the predictions at test-time.
Higher entropy means less confidence in predictions and more inaccurate classifications leading 
to poorer Dice coefficients.

<figure class="figure">
  <img src="../assets/images/posts/2020-07-02-uncertainty-estimation/predictive_uncertainty.png" class="figure-img img-fluid rounded" alt="A generic square placeholder image with rounded corners in a figure.">
  <figcaption class="figure-caption">A caption for the above image.</figcaption>
</figure>


#### Out-of-distribution Detection
Another important aspect of uncertainty estimation  is the ability of a predictive model to distinguish 
*in-distribution* test examples (i.e. those similar to the training data) from 
*out-of-distribution* (OOD) test examples (i.e. those that do not fit the distribution of the training data).
In medical imaging applications as deep networks are often [sensitive to *domain shift*](https://arxiv.org/abs/1702.07841).
For instance, networks trained on one MRI protocol often do not perform satisfactorily on images obtained with slightly different parameters or OOD test images.
Hence, in the face of an OOD sample, an ideal model knows
and announces *"I do not know"* and seeks human intervention, if possible, instead of a silent failure.

In our experiments we use two public datasets: PROSTATEx ([images](),
 [labels]()) and [PROMISE12](). 
PROSTATEx dataset was used for training, while PROMISE12 dataset was set aside for test only.
PROSTATEx dataset was collected in a single institute and all the images were acquired using [phased-array coils]().
PROMISE12 dataset is a heterogeneous multi-institutional dataset acquired using different 
MR scanners and acquisition parameters.
As a result of domain shift, models trained on phase-array coils do not work well on images acquired with
endorectal coils and sometimes they fail drastically.  
We observe that calibrated FCNs have the potential to detect OOD samples.


<figure class="figure">
  <img src="../assets/images/posts/2020-07-02-uncertainty-estimation/predictive_uncertainty.png" class="figure-img img-fluid rounded" alt="A generic square placeholder image with rounded corners in a figure.">
  <figcaption class="figure-caption">
  The results of inference are shown for two test examples imaged with: (a) phased-array coil (in-distribution example), and (b) endorectal coil (out-of-distribution example).
  The first column shows T2-weighted MRI images with the prostate gland boundary drawn by an expert (white line).
  The second column shows the MRI overlaid with uncalibrated segmentation predictions of an FCN trained with Dice loss.
  The third column shows the calibrated segmentation predictions of an ensemble of FCNs trained with Dice loss.
  The fourth column shows the histogram of the calibrated class probabilities over the predicted prostate segment of the whole volume. 
  Note that the bottom row has a much wider distribution compared to the top row, indicating that this is an out of distribution example. 
  </figcaption>
</figure>

<!-- ##### Segmentation Quality -->

##### Conclusions
- Model ensembling is effective not only for improving segmentation quality but also for confidence calibration.
- Ensembling significantly improves the calibration qualities of FCNs trained with Dice Loss and BN.
- We observed that average entropy of the predicted segments correlates with Dice score. 
Hence, it can be used as an effective metric for predicting the test-time performance when the ground-truth is unknown.
- Well-calibrated models can detect out-of-distribution examples and predict failures.

<div class="share-page">
    Share this post on &rarr;
    <a href="https://twitter.com/intent/tweet?text={{ page.title }}&url={{ site.url }}{{ page.url }}&via={{ site.twitter_username }}&related={{ site.twitter_username }}" rel="nofollow" target="_blank" title="Share on Twitter">Twitter</a> |
    <a href="https://facebook.com/sharer.php?u={{ site.url }}{{ page.url }}" rel="nofollow" target="_blank" title="Share on Facebook">Facebook</a> |
    <a href="http://www.linkedin.com/shareArticle?mini=true&url={{ site.url }}{{ page.url }}&title={{ page.title }}&summary=<DESCRIPTION>&source=<DOMAIN>" rel="nofollow" target="_blank" title="Share on LinkedIn">LinkedIn</a> |
    <a href="https://www.linkedin.com/shareArticle?mini=true&url={{ page.url | absolute_url | url_encode }}">LinkedIn</a>
</div>

