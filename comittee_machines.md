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
- Let our learning data be given by 
	$$
	\mathcal{L}=\{(\boldsymbol{x_i}, y_i), i=1,2,\cdots,n\},
	$$
where $\{y_i\}$ are either continuous variables (regression) or unordered labels (classification). We then create $B$ bootstrap samples by sampling $n$-times with replacement from $\mathcal{L}$ and get 
	$$
	\mathcal{L}^{*b}=\{(\boldsymbol{x_i}^{*b}, y_i^{*b}), i=1,2,\cdots,n\}, \qquad b=1,2,\cdots,B.
	$$
Each data point will be chosen each time with a probability of $p=1/n$, because we sample with replacement the resulting distribution of $\mathcal{L}^{*b}$ is the same as in $\mathcal{L}$. (PROOF?)

#### Bagging Tree-Based Classifiers
- Let $y_i \in \{1,\cdots,K\}$. From the $b$th bootstrap sample $\mathcal{L}^{*b}$ we train a classification tree $\mathcal{T}^{*b}$. For a new data point $\boldsymbol{x}$ we predict their class by letting all $B$ clasification trees vote (*majority-vote rule*).
- Because we sample with replacement to get $\mathcal{T}^{*b}$ about $37\%$ of the original data will not occur in $\mathcal{L}^{*b}$. (PROOF) The observations in $\mathcal{L}^ - \mathcal{L}^{*b}:= \left( (\boldsymbol{x}, y) \in \mathcal{L} \text{ and } \boldsymbol{x}, y) \in \mathcal{L}^{*b} \right)$ are called *out-of-bag obeservations (OBB)*. Using the OOB to evaluate the generalitation error is equivalent to using a independent sample with the same distribution as  $\mathcal{L}$. This way there is no need to shorten the available data by splitting it into a training set and a validation or test set  and also makes cross-validation unnecessary.
- 
	
