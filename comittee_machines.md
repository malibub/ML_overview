# Committee Machines

[Source](https://link.springer.com/book/10.1007/978-0-387-78189-1)

- reducing bias or variance (and hence generalization error) is an important topic in ML but still the definitions of both are not uniquely defined for classification (as it is for regression)
	- High variance and low bias lead to overfitting. In this case a classification method is said to be *unstable* if small changes in the learning set include major changes in the classifier. Decision trees or neuronal nets are therefore by definition unstable.
	- High bias underfits the data. (E.g. linear discriminant analysis)
- Instabilty of a classifier can be used though to improve accuracy, e.g. by using bagging or boosting methods
	- Bagging: random and independent drawings from learning data, reduces variance
	- Boosting: deterministic,  successively reweighting the learning set, current weights depend on misclassification history, reduces bias
- By perturbing the learning data an ensemble of classifiers is created which then are combined to one. This process is called *committee machine* or *ensemble machine*

## Bagging 
- acronym for *bootstraping aggregating*
- first algorithm that successfully improved classification performance compared to the single classifier
- Bagging ist most successful when the predictor is unstable, otherwise it can even weaken performance.
- Let our learning data be given by $$ \mathcal{L}=\left\{(\boldsymbol{x_i}, y_i), i=1,2,\cdots,n\right\}	$$
where $\{y_i\}$ are either continuous variables (regression) or unordered labels (classification). We then create $B$ bootstrap samples by sampling $n$-times with replacement from $\mathcal{L}$ and get $$ \mathcal{L}^{*b}=\{(\boldsymbol{x_i}^{*b}, y_i^{*b}), i=1,2,\cdots,n\}, \qquad b=1,2,\cdots,B. $$ Each data point will be chosen each time with a probability of $p=1/n$, because we sample with replacement the resulting distribut^on of $\mathcal{L}^{*b}$ is the same as in $\mathcal{L}$. (PROOF?)

#### Bagging Tree-Based Classifiers
- Let $y_i \in \{1,\cdots,K\}$. From the $b$th bootstrap sample $\mathcal{L}^{*b}$ we train a classification tree $\mathcal{T}^{*b}$. For a new data point $\boldsymbol{x}$ we predict their class by letting all $B$ clasification trees vote (*majority-vote rule*).
- Because we sample with replacement to get $\mathcal{T}^{*b}$ about $37\%$ of the original data will not occur in $\mathcal{L}^{*b}$. (PROOF) The observations in $\mathcal{L}^ - \mathcal{L}^{*b}:= \left\{ (\boldsymbol{x}, y) \in \mathcal{L} \text{ and } (\boldsymbol{x}, y) \in \mathcal{L}^{*b} \right)\}$ are called *out-of-bag obeservations (OBB)*. Using the OOB to evaluate the generalitation error is equivalent to using a independent sample with the same distribution as  $\mathcal{L}$. This way there is no need to shorten the available data by splitting it into a training set and a validation or test set  and also makes cross-validation unnecessary.
-  Now let $(\boldsymbol{x_i}, y_i) \notin \mathcal{L}^{*b}$ for $n_i$ of the $B$ bootstrap samples. We drop down $\boldsymbol{x_i}$ their corresponding trees $\mathcal{T}^{*b}$ to each vote for one of the $K$ classes. 
- The results are summarized in the $K$-sized vector $\boldsymbol{\hat{p}}(\boldsymbol{x_i}):= (\hat{p}_1(\boldsymbol{x_i}), \cdots,\hat{p}_K(\boldsymbol{x_i}))^T$, where $\hat{p}_k(\boldsymbol{x_i})$ denotes the proportion of the votes for class $k$ as an estimate of the true probability $p(\Pi_k \mid \boldsymbol{x_i})=P(\boldsymbol{X}\in \Pi_k \mid \boldsymbol{X}=\boldsymbol{x_i})$.
- The *OBB classifier* then is $C_{bag}(\boldsymbol{x_i})\qquad =\qquad arg \max_{k} \quad \hat{p}_k(\boldsymbol{x_i})$, so the classifier decides for the calss with the most votes.
- The *misclassification rate* now is given by $ PE_{bag} := \frac{1}{n} \sum_{i=1}^n \boldsymbol{1}_{[C_{bag}(\boldsymbol{x_i}) \ne y_i]}$, the proportion of misclassifications over all observations in $\mathcal{L}$. (unbaised since we use classifiers that do not know the true class of $\boldsymbol{x_i}$).

#### Bagging Regression-Tree Predictors
- In this case $y_i \in \mathbb{R}$, so now instead of voting for the most popular calss we predict the response $y$ by averaging the predicted values ovtained from out $B$ classifiers.
- Dropping down $\boldsymbol{x}$ all $B$ regression trees $T^{*b}$ we otain the bagged estimate of $y$ as $$ \hat{\mu}_{bag}(\boldsymbol{x})=\frac{1}{B}\sum_{b=1}^B \hat{\mu}^{*b}(\boldsymbol{x}). $$ 
- While the *OBB regression estimate* for $\boldsymbol{x_i}$ from the learning data is given by $$\hat{\mu}_{bag}(\boldsymbol{x_i})=\frac{1}{n_i}\sum_{b\in \mathcal{N_i}} \hat{\mu}^{*b}(\boldsymbol{x_i}),$$ where $\mathcal{N_i}$ is the set of the indices of the $n_i$ bootstrap samples that do not contain $\boldsymbol{x_i}$.
- The *OBB error rate* is given by $$PE_{bag} \quad = \quad \frac{1}{n}\sum_{i=1}^n \left( y_i- \hat{\mu}_{bag}(\boldsymbol{x_i}) \right)^2$$ (mean-squared-error of the bagged estimat compares to the true response).


## Boosting 
- The main idea of boostng is to combine $M$ weak classifiers $C_1,\cdots,C_M$, e.g. ones that correctly classify about 50% of the time, to one *boosted classfier* $$C_{\boldsymbol{\alpha}}(\boldsymbol{x})=sgn(f_{\boldsymbol{\alpha}}(\boldsymbol{x})) $$ where $\boldsymbol{\alpha}=(\alpha_1,\cdots,\alpha_M)^T$ is a vector containing constants and $$f_{\boldsymbol{\alpha}}(\boldsymbol{x}) = \sum_{ij=1}^M\left( \frac{\alpha_j}{\sum_{j`}\alpha_{j`}}\right)C_j(\boldsymbol{x}) .$$
- Depending on the complexity of the classification problem one-level to three-level desicion trees are used as the weak classifiers. Each of these weak classifiers could for example take a distinct characteristic / explanatory variable.
#### ADABoosting (boosting by reweighting)
- ADABoosting is an acronym for *adaptive boosting* and is regarded as the first attempt toward a truly practical boosting method for binray classification and is devised to driving the predition error from the learnign set to $0$. (For two or more classes this is calles ADABoosting.M1)

