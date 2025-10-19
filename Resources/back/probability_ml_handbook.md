# The Handbook of Probability for Machine Learning

This guide breaks down the essential probability theory you need for machine learning, moving from the "what" (the theory) to the "why" (the intuition) and finally to the "how" (the code).

Machine learning is fundamentally about dealing with uncertainty. We never have perfect information about the world, so we use probability to quantify and reason about what we don't know.

Probability is the mathematical language of uncertainty and the foundation of statistical learning.

## Chapter 1: Random Variables (Quantifying Uncertainty)

**What It Is:** A random variable is a variable whose value is determined by chance. It's a function that maps outcomes of a random process to numbers.

**Types:**
- **Discrete Random Variables:** Can only take specific values (e.g., coin flips: 0 or 1, dice rolls: 1-6)
- **Continuous Random Variables:** Can take any value in a range (e.g., height, temperature, model predictions)

**The Math:** We write $X$ for a random variable and $x$ for a specific value it can take.

- $P(X = x)$ means "the probability that $X$ takes the value $x$"
- For discrete variables: $\sum_x P(X = x) = 1$ (all probabilities sum to 1)
- For continuous variables: $\int_{-\infty}^{\infty} p(x) dx = 1$ (total area under the curve is 1)

**Why It Matters:** Every prediction your model makes is a random variable. The input features are random variables. Understanding their behavior is essential for building robust models.

### 💻 Practical Python: Simulating Random Variables

Let's simulate a simple random variable: rolling a fair die.

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Simulate rolling a die 10,000 times
np.random.seed(42)
n_rolls = 10000
dice_rolls = np.random.randint(1, 7, size=n_rolls)  # Values from 1 to 6

# 2. Calculate the empirical probability (frequency)
values, counts = np.unique(dice_rolls, return_counts=True)
probabilities = counts / n_rolls

# 3. Plot the distribution
plt.figure(figsize=(10, 6))
plt.bar(values, probabilities, alpha=0.7, color='steelblue', edgecolor='black')
plt.axhline(y=1/6, color='red', linestyle='--', label='Theoretical probability (1/6)')
plt.xlabel('Die Value')
plt.ylabel('Probability')
plt.title('Empirical Distribution of Dice Rolls')
plt.xticks(values)
plt.ylim(0, 0.25)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()

print(f"Empirical probabilities: {probabilities}")
print(f"Theoretical probability: {1/6:.4f}")
```

Notice how with many samples, the empirical frequencies converge to the theoretical probability of 1/6. This is the **Law of Large Numbers**.

## Chapter 2: Probability Distributions (The Shape of Uncertainty)

**What It Is:** A probability distribution describes how probability is spread across all possible values of a random variable.

**Key Distributions for ML:**

### Bernoulli Distribution (Single Binary Event)
Models a single yes/no experiment (e.g., will this email be spam?).
- Parameter: $p$ (probability of success)
- $P(X = 1) = p$ and $P(X = 0) = 1 - p$

### Binomial Distribution (Multiple Binary Events)
Models the number of successes in $n$ independent Bernoulli trials.
- Parameters: $n$ (number of trials), $p$ (probability of success)
- Used in: Classification metrics, A/B testing

### Normal (Gaussian) Distribution (The Bell Curve)
The most important distribution in ML. Many natural phenomena follow this.
- Parameters: $\mu$ (mean), $\sigma^2$ (variance)
- $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
- Used in: Linear regression assumptions, neural network initialization, Gaussian processes

### Uniform Distribution (Equal Probability)
Every value in a range is equally likely.
- Used in: Random weight initialization, data augmentation

**Why It Matters:** 
- Understanding data distributions helps you choose the right model
- Many ML algorithms make assumptions about the distribution of your data
- Loss functions are often derived from probability distributions

### 💻 Practical Python: Visualizing Key Distributions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Bernoulli Distribution
p = 0.7
x_bernoulli = [0, 1]
pmf_bernoulli = [1-p, p]
axes[0, 0].bar(x_bernoulli, pmf_bernoulli, color='steelblue', edgecolor='black')
axes[0, 0].set_title(f'Bernoulli Distribution (p={p})')
axes[0, 0].set_xlabel('Outcome')
axes[0, 0].set_ylabel('Probability')
axes[0, 0].set_xticks([0, 1])

# 2. Binomial Distribution
n, p = 20, 0.5
x_binomial = np.arange(0, n+1)
pmf_binomial = stats.binom.pmf(x_binomial, n, p)
axes[0, 1].bar(x_binomial, pmf_binomial, color='coral', edgecolor='black')
axes[0, 1].set_title(f'Binomial Distribution (n={n}, p={p})')
axes[0, 1].set_xlabel('Number of Successes')
axes[0, 1].set_ylabel('Probability')

# 3. Normal Distribution
mu, sigma = 0, 1
x_normal = np.linspace(-4, 4, 1000)
pdf_normal = stats.norm.pdf(x_normal, mu, sigma)
axes[1, 0].plot(x_normal, pdf_normal, color='green', linewidth=2)
axes[1, 0].fill_between(x_normal, pdf_normal, alpha=0.3, color='green')
axes[1, 0].set_title(f'Normal Distribution (μ={mu}, σ={sigma})')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Probability Density')
axes[1, 0].grid(alpha=0.3)

# 4. Uniform Distribution
a, b = 0, 10
x_uniform = np.linspace(a-1, b+1, 1000)
pdf_uniform = stats.uniform.pdf(x_uniform, a, b-a)
axes[1, 1].plot(x_uniform, pdf_uniform, color='purple', linewidth=2)
axes[1, 1].fill_between(x_uniform, pdf_uniform, alpha=0.3, color='purple')
axes[1, 1].set_title(f'Uniform Distribution (a={a}, b={b})')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Probability Density')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Chapter 3: Expected Value and Variance (Summarizing Distributions)

**What They Are:** Instead of keeping track of an entire distribution, we often summarize it with two key numbers.

### Expected Value (Mean)
The "average" value we expect if we sample many times.

**For discrete variables:**
$$E[X] = \sum_x x \cdot P(X = x)$$

**For continuous variables:**
$$E[X] = \int_{-\infty}^{\infty} x \cdot p(x) dx$$

**Intuition:** It's the "center of mass" of the distribution.

### Variance
Measures how spread out the distribution is from the mean.

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

**Standard Deviation:** $\sigma = \sqrt{\text{Var}(X)}$ (in the same units as $X$)

**Why It Matters:**
- The mean is your model's point prediction
- The variance quantifies uncertainty in your prediction
- Many ML algorithms try to minimize variance (prevent overfitting)
- Regularization techniques control the variance of model parameters

### 💻 Practical Python: Computing Moments

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate data from a normal distribution
np.random.seed(42)
mu_true, sigma_true = 5, 2
data = np.random.normal(mu_true, sigma_true, 1000)

# 2. Compute sample statistics
mean_sample = np.mean(data)
var_sample = np.var(data, ddof=1)  # ddof=1 for unbiased estimator
std_sample = np.std(data, ddof=1)

# 3. Visualize
plt.figure(figsize=(12, 5))

# Plot histogram
plt.subplot(1, 2, 1)
plt.hist(data, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
plt.axvline(mean_sample, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_sample:.2f}')
plt.axvline(mean_sample - std_sample, color='orange', linestyle='--', linewidth=2, label=f'±1 Std Dev')
plt.axvline(mean_sample + std_sample, color='orange', linestyle='--', linewidth=2)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Distribution with Mean and Standard Deviation')
plt.legend()
plt.grid(alpha=0.3)

# Plot cumulative distribution
plt.subplot(1, 2, 2)
sorted_data = np.sort(data)
cumulative = np.arange(1, len(data) + 1) / len(data)
plt.plot(sorted_data, cumulative, linewidth=2, color='green')
plt.axvline(mean_sample, color='red', linestyle='--', linewidth=2, label=f'Mean (50th percentile)')
plt.axhline(0.5, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution Function (CDF)')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"True parameters: μ={mu_true}, σ={sigma_true}")
print(f"Sample estimates: μ̂={mean_sample:.2f}, σ̂={std_sample:.2f}")
print(f"Variance: {var_sample:.2f}")
```

## Chapter 4: Joint and Conditional Probability (Relationships Between Variables)

**What They Are:** Real-world problems involve multiple random variables that are related to each other.

### Joint Probability
The probability that two events happen together.

$$P(X = x, Y = y)$$ or $$p(x, y)$$

**Example:** $P(\text{Email is Spam AND contains word "free"})$

### Conditional Probability
The probability of one event given that another has occurred.

$$P(Y = y | X = x) = \frac{P(X = x, Y = y)}{P(X = x)}$$

**Intuition:** "What's the probability of $Y$ in the restricted universe where $X$ has already happened?"

**Example:** $P(\text{Spam | contains "free"})$ - much higher than $P(\text{Spam})$ alone!

### Independence
Two variables are independent if knowing one tells you nothing about the other.

$$P(X, Y) = P(X) \cdot P(Y)$$
$$P(Y | X) = P(Y)$$

**Why It Matters:**
- **Feature relationships:** Understanding which features are correlated helps feature engineering
- **Naive Bayes:** Assumes features are conditionally independent (the "naive" assumption)
- **Causal inference:** Distinguishes correlation from causation
- **Generative models:** Model joint distributions to generate new data

### 💻 Practical Python: Joint and Conditional Distributions

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create synthetic data: height and weight (correlated)
np.random.seed(42)
n_samples = 1000

# Height in cm (mean=170, std=10)
height = np.random.normal(170, 10, n_samples)

# Weight depends on height (correlation)
weight = 0.9 * height - 100 + np.random.normal(0, 5, n_samples)

# 2. Create 2D histogram for joint distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Joint distribution
axes[0, 0].scatter(height, weight, alpha=0.3, s=10)
axes[0, 0].set_xlabel('Height (cm)')
axes[0, 0].set_ylabel('Weight (kg)')
axes[0, 0].set_title('Joint Distribution: P(Height, Weight)')
axes[0, 0].grid(alpha=0.3)

# 2D histogram
h, xedges, yedges = np.histogram2d(height, weight, bins=30)
axes[0, 1].imshow(h.T, origin='lower', aspect='auto', cmap='viridis',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
axes[0, 1].set_xlabel('Height (cm)')
axes[0, 1].set_ylabel('Weight (kg)')
axes[0, 1].set_title('Joint Probability Density')
plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], label='Density')

# Marginal distribution of height: P(Height)
axes[1, 0].hist(height, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[1, 0].set_xlabel('Height (cm)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Marginal Distribution: P(Height)')
axes[1, 0].grid(alpha=0.3)

# Conditional distribution: P(Weight | Height ≈ 170)
# Select weights for people with height between 169 and 171
condition_mask = (height >= 169) & (height <= 171)
weight_conditional = weight[condition_mask]
axes[1, 1].hist(weight_conditional, bins=20, density=True, alpha=0.7, color='coral', edgecolor='black')
axes[1, 1].set_xlabel('Weight (kg)')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Conditional Distribution: P(Weight | Height ≈ 170)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate correlation
correlation = np.corrcoef(height, weight)[0, 1]
print(f"Correlation between height and weight: {correlation:.3f}")
```

## Chapter 5: Bayes' Theorem (The Foundation of Learning)

**What It Is:** Bayes' Theorem is arguably the most important equation in machine learning. It tells us how to update our beliefs in light of new evidence.

$$P(H | E) = \frac{P(E | H) \cdot P(H)}{P(E)}$$

**Translation:**
- $P(H | E)$: **Posterior** - probability of hypothesis $H$ after seeing evidence $E$
- $P(E | H)$: **Likelihood** - probability of seeing evidence $E$ if hypothesis $H$ is true
- $P(H)$: **Prior** - probability of hypothesis $H$ before seeing evidence
- $P(E)$: **Evidence** - total probability of seeing evidence $E$

**The Intuition (Medical Diagnosis):**

You test positive for a rare disease. Should you panic?

- **Prior:** $P(\text{Disease}) = 0.001$ (0.1% of people have it)
- **Likelihood:** $P(\text{Positive} | \text{Disease}) = 0.99$ (test is 99% accurate)
- **False positive rate:** $P(\text{Positive} | \text{No Disease}) = 0.05$ (5% false positive rate)

What's $P(\text{Disease} | \text{Positive})$?

$$P(\text{Disease} | \text{Positive}) = \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.05 \times 0.999} = \frac{0.00099}{0.05094} \approx 0.019$$

**Only 1.9%!** Most positives are false positives because the disease is so rare.

**Why It Matters:**
- **Bayesian inference:** The principled way to incorporate prior knowledge
- **Naive Bayes classifier:** Fast and effective for text classification
- **Bayesian neural networks:** Quantify uncertainty in predictions
- **A/B testing:** Update beliefs about which variant is better
- **Spam filtering:** $P(\text{Spam} | \text{words})$

### 💻 Practical Python: Bayesian Spam Classifier

```python
import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.word_counts = {'spam': defaultdict(int), 'ham': defaultdict(int)}
        self.class_counts = {'spam': 0, 'ham': 0}
        self.vocab = set()
    
    def fit(self, emails, labels):
        """Train the classifier"""
        for email, label in zip(emails, labels):
            self.class_counts[label] += 1
            words = email.lower().split()
            for word in words:
                self.word_counts[label][word] += 1
                self.vocab.add(word)
    
    def predict(self, email):
        """Predict if email is spam or ham"""
        words = email.lower().split()
        
        # Calculate log probabilities to avoid underflow
        log_prob_spam = np.log(self.class_counts['spam'] / sum(self.class_counts.values()))
        log_prob_ham = np.log(self.class_counts['ham'] / sum(self.class_counts.values()))
        
        # Calculate P(words | spam) and P(words | ham)
        for word in words:
            # Laplace smoothing: add 1 to avoid zero probabilities
            spam_count = self.word_counts['spam'][word] + 1
            ham_count = self.word_counts['ham'][word] + 1
            spam_total = sum(self.word_counts['spam'].values()) + len(self.vocab)
            ham_total = sum(self.word_counts['ham'].values()) + len(self.vocab)
            
            log_prob_spam += np.log(spam_count / spam_total)
            log_prob_ham += np.log(ham_count / ham_total)
        
        # Return the class with higher probability
        return 'spam' if log_prob_spam > log_prob_ham else 'ham'
    
    def predict_proba(self, email):
        """Return probability of spam"""
        words = email.lower().split()
        
        log_prob_spam = np.log(self.class_counts['spam'] / sum(self.class_counts.values()))
        log_prob_ham = np.log(self.class_counts['ham'] / sum(self.class_counts.values()))
        
        for word in words:
            spam_count = self.word_counts['spam'][word] + 1
            ham_count = self.word_counts['ham'][word] + 1
            spam_total = sum(self.word_counts['spam'].values()) + len(self.vocab)
            ham_total = sum(self.word_counts['ham'].values()) + len(self.vocab)
            
            log_prob_spam += np.log(spam_count / spam_total)
            log_prob_ham += np.log(ham_count / ham_total)
        
        # Convert back from log space
        prob_spam = np.exp(log_prob_spam)
        prob_ham = np.exp(log_prob_ham)
        
        return prob_spam / (prob_spam + prob_ham)

# Example usage
train_emails = [
    "free money now click here", 
    "meeting tomorrow at 3pm",
    "win a free iphone today",
    "project deadline reminder",
    "congratulations you won",
    "lunch with team next week"
]
train_labels = ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']

# Train classifier
classifier = NaiveBayesClassifier()
classifier.fit(train_emails, train_labels)

# Test
test_emails = [
    "free prize winner",
    "meeting notes attached",
    "win lottery now"
]

print("Predictions:")
for email in test_emails:
    prediction = classifier.predict(email)
    probability = classifier.predict_proba(email)
    print(f"Email: '{email}'")
    print(f"  Predicted: {prediction} (P(spam) = {probability:.3f})\n")
```

## Chapter 6: Maximum Likelihood Estimation (Learning from Data)

**What It Is:** Given some data, what are the most likely parameters of the distribution that generated it?

**The Setup:** We have data $D = \{x_1, x_2, ..., x_n\}$ and we assume it comes from a distribution with parameters $\theta$.

**The Goal:** Find $\theta$ that maximizes the likelihood of observing our data.

$$\theta^* = \arg\max_\theta P(D | \theta) = \arg\max_\theta \prod_{i=1}^n P(x_i | \theta)$$

**In Practice:** We maximize the log-likelihood (easier math, same answer):

$$\theta^* = \arg\max_\theta \log P(D | \theta) = \arg\max_\theta \sum_{i=1}^n \log P(x_i | \theta)$$

**Example: Coin Flips**

You flip a coin 100 times and get 60 heads. What's the most likely value of $p$ (probability of heads)?

The likelihood is:
$$L(p) = \binom{100}{60} p^{60} (1-p)^{40}$$

Taking the log and maximizing:
$$\frac{d}{dp} \log L(p) = \frac{60}{p} - \frac{40}{1-p} = 0$$

Solving: $p^* = 0.6$ (exactly the observed frequency!)

**Why It Matters:**
- **Linear regression:** Minimizing squared error is equivalent to MLE assuming Gaussian noise
- **Logistic regression:** MLE with Bernoulli distribution
- **Neural networks:** Cross-entropy loss comes from MLE
- **It connects probability to optimization:** Finding parameters = minimizing negative log-likelihood

### 💻 Practical Python: MLE for Gaussian Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Generate data from unknown distribution
np.random.seed(42)
true_mu, true_sigma = 5, 2
data = np.random.normal(true_mu, true_sigma, 100)

# 2. Try different parameter values
mu_range = np.linspace(3, 7, 100)
sigma_range = np.linspace(0.5, 4, 100)

# 3. Calculate log-likelihood for each parameter combination
log_likelihoods = np.zeros((len(mu_range), len(sigma_range)))

for i, mu in enumerate(mu_range):
    for j, sigma in enumerate(sigma_range):
        # Log-likelihood = sum of log(pdf(x | mu, sigma))
        log_likelihoods[i, j] = np.sum(stats.norm.logpdf(data, mu, sigma))

# 4. Find MLE estimates
max_idx = np.unravel_index(np.argmax(log_likelihoods), log_likelihoods.shape)
mle_mu = mu_range[max_idx[0]]
mle_sigma = sigma_range[max_idx[1]]

# 5. Compare with analytical solution (sample mean and std)
analytical_mu = np.mean(data)
analytical_sigma = np.std(data, ddof=1)

# 6. Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot log-likelihood surface
axes[0].contourf(sigma_range, mu_range, log_likelihoods, levels=30, cmap='viridis')
axes[0].scatter(mle_sigma, mle_mu, color='red', s=200, marker='*', 
                edgecolor='white', linewidth=2, label=f'MLE: μ={mle_mu:.2f}, σ={mle_sigma:.2f}')
axes[0].scatter(true_sigma, true_mu, color='lime', s=200, marker='x',
                linewidth=3, label=f'True: μ={true_mu}, σ={true_sigma}')
axes[0].set_xlabel('σ (Standard Deviation)')
axes[0].set_ylabel('μ (Mean)')
axes[0].set_title('Log-Likelihood Surface')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot data with fitted distribution
axes[1].hist(data, bins=20, density=True, alpha=0.6, color='steelblue', 
             edgecolor='black', label='Observed Data')

x_plot = np.linspace(data.min(), data.max(), 1000)
axes[1].plot(x_plot, stats.norm.pdf(x_plot, mle_mu, mle_sigma), 
             'r-', linewidth=2, label=f'MLE Fit: N({mle_mu:.2f}, {mle_sigma:.2f}²)')
axes[1].plot(x_plot, stats.norm.pdf(x_plot, true_mu, true_sigma), 
             'g--', linewidth=2, label=f'True: N({true_mu}, {true_sigma}²)')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Density')
axes[1].set_title('Data and Fitted Distribution')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"True parameters: μ={true_mu}, σ={true_sigma}")
print(f"MLE estimates: μ={mle_mu:.2f}, σ={mle_sigma:.2f}")
print(f"Analytical (sample stats): μ={analytical_mu:.2f}, σ={analytical_sigma:.2f}")
```

## Chapter 7: Information Theory Basics (Measuring Surprise)

**What It Is:** Information theory quantifies uncertainty, information, and surprise. It's the foundation of many ML concepts.

### Entropy (Average Surprise)

How uncertain/random is a distribution?

$$H(X) = -\sum_x P(x) \log_2 P(x)$$

**Intuition:**
- **Low entropy:** Distribution is concentrated (e.g., coin with $p=0.99$) - predictable
- **High entropy:** Distribution is spread out (e.g., fair coin) - unpredictable

**For a fair coin:** $H = -0.5 \log_2(0.5) - 0.5 \log_2(0.5) = 1$ bit

**For a biased coin** ($p=0.99$): $H = -0.99 \log_2(0.99) - 0.01 \log_2(0.01) = 0.08$ bits

### Cross-Entropy (Surprise Under Wrong Model)

If we think the distribution is $Q$ but it's actually $P$:

$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

**Why It Matters:** This is the **cross-entropy loss** in classification!
- $P$ = true labels (one-hot encoded)
- $Q$ = model predictions (probabilities)
- Minimizing cross-entropy = making model predictions match true distribution

### KL Divergence (Distance Between Distributions)

How different are two distributions?

$$D_{KL}(P || Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = H(P, Q) - H(P)$$

**Properties:**
- Always ≥ 0
- = 0 only when $P = Q$
- NOT symmetric: $D_{KL}(P || Q) \neq D_{KL}(Q || P)$

**Why It Matters:**
- **Variational autoencoders:** Minimize KL between learned distribution and prior
- **Model comparison:** Measure how well model approximates true distribution
- **Regularization:** KL divergence as a penalty term

### 💻 Practical Python: Entropy and Cross-Entropy

```python
import numpy as np
import matplotlib.pyplot as plt

def entropy(probs):
    """Calculate entropy (in bits)"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Remove zeros to avoid log(0)
    return -np.sum(probs * np.log2(probs))

def cross_entropy(true_probs, pred_probs):
    """Calculate cross-entropy"""
    true_probs = np.array(true_probs)
    pred_probs = np.array(pred_probs)
    pred_probs = np.clip(pred_probs, 1e-10, 1)  # Avoid log(0)
    return -np.sum(true_probs * np.log2(pred_probs))

def kl_divergence(p, q):
    """Calculate KL divergence from Q to P"""
    p = np.array(p)
    q = np.array(q)
    q = np.clip(q, 1e-10, 1)  # Avoid division by zero
    return np.sum(p * np.log2(p / q))

# 1. Visualize entropy for a binary variable
p_values = np.linspace(0.01, 0.99, 100)
entropies = [entropy([p, 1-p]) for p in p_values]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(p_values, entropies, linewidth=2, color='steelblue')
axes[0, 0].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Maximum entropy at p=0.5')
axes[0, 0].set_xlabel('P(X=1)')
axes[0, 0].set_ylabel('Entropy (bits)')
axes[0, 0].set_title('Entropy of Binary Variable')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Cross-entropy for classification
# True label: class 0 (one-hot: [1, 0, 0])
true_label = np.array([1, 0, 0])

# Model predictions (varying confidence in correct class)
confidences = np.linspace(0.1, 0.99, 100)
ce_losses = []

for conf in confidences:
    pred = np.array([conf, (1-conf)/2, (1-conf)/2])
    ce_losses.append(cross_entropy(true_label, pred))

axes[0, 1].plot(confidences, ce_losses, linewidth=2, color='coral')
axes[0, 1].set_xlabel('Model Confidence in Correct Class')
axes[0, 1].set_ylabel('Cross-Entropy Loss')
axes[0, 1].set_title('Cross-Entropy vs Model Confidence')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].invert_xaxis()  # Higher confidence = lower loss

# 3. Compare distributions with KL divergence
categories = ['A', 'B', 'C', 'D']
p_true = np.array([0.4, 0.3, 0.2, 0.1])
q_good = np.array([0.35, 0.35, 0.2, 0.1])   # Close to P
q_bad = np.array([0.1, 0.2, 0.3, 0.4])      # Far from P

x = np.arange(len(categories))
width = 0.25

axes[1, 0].bar(x - width, p_true, width, label='True P', color='green', alpha=0.7)
axes[1, 0].bar(x, q_good, width, label=f'Good Q (KL={kl_divergence(p_true, q_good):.3f})', 
               color='blue', alpha=0.7)
axes[1, 0].bar(x + width, q_bad, width, label=f'Bad Q (KL={kl_divergence(p_true, q_bad):.3f})', 
               color='red', alpha=0.7)
axes[1, 0].set_xlabel('Category')
axes[1, 0].set_ylabel('Probability')
axes[1, 0].set_title('KL Divergence: Comparing Distributions')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(categories)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# 4. Information gain in decision trees
# Parent node (mixed class)
parent = np.array([0.5, 0.5])
# After split: left child (pure), right child (mixed)
left_child = np.array([0.9, 0.1])
right_child = np.array([0.2, 0.8])

parent_entropy = entropy(parent)
left_entropy = entropy(left_child)
right_entropy = entropy(right_child)

# Weighted average of children (assume 50-50 split)
weighted_child_entropy = 0.5 * left_entropy + 0.5 * right_entropy
information_gain = parent_entropy - weighted_child_entropy

nodes = ['Parent\n(mixed)', 'Left Child\n(mostly class 0)', 'Right Child\n(mostly class 1)']
entropies = [parent_entropy, left_entropy, right_entropy]
colors = ['orange', 'lightblue', 'lightgreen']

axes[1, 1].bar(nodes, entropies, color=colors, edgecolor='black', linewidth=2)
axes[1, 1].set_ylabel('Entropy (bits)')
axes[1, 1].set_title(f'Information Gain in Decision Tree\nGain = {information_gain:.3f} bits')
axes[1, 1].grid(alpha=0.3, axis='y')
axes[1, 1].axhline(parent_entropy, color='red', linestyle='--', 
                    alpha=0.5, label='Parent entropy')

plt.tight_layout()
plt.show()

print("Entropy Examples:")
print(f"Fair coin: {entropy([0.5, 0.5]):.3f} bits")
print(f"Biased coin (0.9): {entropy([0.9, 0.1]):.3f} bits")
print(f"Uniform over 8 outcomes: {entropy([1/8]*8):.3f} bits")
print(f"\nInformation Gain: {information_gain:.3f} bits")
```

## Chapter 8: The Central Limit Theorem (Why the Normal Distribution is Everywhere)

**What It Is:** One of the most important theorems in probability. It explains why the normal distribution appears everywhere in nature and ML.

**The Statement:** When you add together many independent random variables (regardless of their individual distributions), their sum approximates a normal distribution.

$\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{n \to \infty} N(0, 1)$

Where:
- $\bar{X}$ is the sample mean
- $\mu$ is the true mean
- $\sigma$ is the true standard deviation
- $n$ is the sample size

**The Intuition:** Individual events might be weird and random, but when you aggregate many of them, the result becomes predictable and bell-shaped.

**Why It Matters:**
- **Error distributions:** Prediction errors are often sums of many small effects → Gaussian
- **Gradient descent:** With large batches, gradient estimates are approximately normal
- **Confidence intervals:** We can make probabilistic statements about our estimates
- **Hypothesis testing:** Statistical tests rely on CLT for validity
- **Neural networks:** Initialization schemes assume activations will be approximately normal

### 💻 Practical Python: Demonstrating the CLT

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# 1. Start with a non-normal distribution: Uniform
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Parameters
n_samples = 1000
sample_sizes = [1, 2, 5, 10, 30, 100]

# Original distribution: Uniform(0, 1)
axes[0, 0].hist(np.random.uniform(0, 1, 10000), bins=50, density=True, 
                alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Original: Uniform Distribution')
axes[0, 0].set_ylabel('Density')
axes[0, 0].grid(alpha=0.3)

# Show how sample means converge to normal as n increases
for idx, n in enumerate(sample_sizes):
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    
    # Generate many sample means
    sample_means = []
    for _ in range(n_samples):
        sample = np.random.uniform(0, 1, n)
        sample_means.append(np.mean(sample))
    
    sample_means = np.array(sample_means)
    
    # Plot histogram
    axes[row, col].hist(sample_means, bins=40, density=True, alpha=0.7, 
                        color='coral', edgecolor='black', label='Sample means')
    
    # Overlay theoretical normal distribution
    mu = 0.5  # Mean of Uniform(0,1)
    sigma = np.sqrt(1/12)  # Std of Uniform(0,1)
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    theoretical = stats.norm.pdf(x, mu, sigma/np.sqrt(n))
    axes[row, col].plot(x, theoretical, 'g-', linewidth=2, label='Theoretical Normal')
    
    axes[row, col].set_title(f'Sample Size n={n}')
    axes[row, col].set_ylabel('Density')
    axes[row, col].legend()
    axes[row, col].grid(alpha=0.3)

axes[0, 1].text(0.5, 0.5, 'Central Limit Theorem:\nMeans converge to\nNormal Distribution', 
                ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[0, 1].axis('off')

axes[0, 2].axis('off')

plt.tight_layout()
plt.show()

# 2. Demonstrate with different starting distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

distributions = {
    'Exponential': lambda size: np.random.exponential(1, size),
    'Binomial': lambda size: np.random.binomial(10, 0.5, size),
    'Chi-squared': lambda size: np.random.chisquare(2, size),
    'Poisson': lambda size: np.random.poisson(3, size)
}

n = 30  # Sample size for means

for idx, (name, dist_func) in enumerate(distributions.items()):
    # Original distribution
    original = dist_func(10000)
    axes[0, idx].hist(original, bins=40, density=True, alpha=0.7, 
                      color='steelblue', edgecolor='black')
    axes[0, idx].set_title(f'Original: {name}')
    axes[0, idx].set_ylabel('Density')
    axes[0, idx].grid(alpha=0.3)
    
    # Sample means distribution
    means = [np.mean(dist_func(n)) for _ in range(1000)]
    axes[1, idx].hist(means, bins=40, density=True, alpha=0.7, 
                      color='coral', edgecolor='black')
    
    # Overlay normal
    x = np.linspace(min(means), max(means), 100)
    normal = stats.norm.pdf(x, np.mean(means), np.std(means))
    axes[1, idx].plot(x, normal, 'g-', linewidth=2)
    axes[1, idx].set_title(f'Means (n={n}): Normal!')
    axes[1, idx].set_ylabel('Density')
    axes[1, idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Chapter 9: Sampling and Monte Carlo Methods (When Math is Too Hard)

**What It Is:** Sometimes calculating probabilities analytically is impossible. Instead, we simulate!

**The Core Idea:** If you can't solve it exactly, approximate it by generating many random samples.

### Monte Carlo Integration

Want to calculate $E[f(X)]$ where $X \sim p(x)$?

Instead of solving: $\int f(x) p(x) dx$

Just sample: $E[f(X)] \approx \frac{1}{N} \sum_{i=1}^N f(x_i)$ where $x_i \sim p(x)$

**Law of Large Numbers guarantees:** As $N \to \infty$, the approximation becomes exact.

### Importance Sampling

What if sampling from $p(x)$ is hard? Sample from a simpler distribution $q(x)$ instead:

$E_p[f(X)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx \approx \frac{1}{N} \sum_{i=1}^N f(x_i) \frac{p(x_i)}{q(x_i)}$

Where $x_i \sim q(x)$.

**Why It Matters:**
- **Reinforcement learning:** Estimate expected rewards
- **Bayesian inference:** Approximate posterior distributions (MCMC)
- **Generative models:** VAEs and GANs use sampling
- **Uncertainty quantification:** Monte Carlo dropout for neural networks
- **Integration:** Calculate complex integrals (e.g., physics simulations)

### 💻 Practical Python: Monte Carlo Estimation

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Example 1: Estimate π using Monte Carlo
def estimate_pi(n_samples):
    """Estimate π by sampling random points in a square"""
    # Generate random points in [0,1] x [0,1]
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)
    
    # Check if points are inside the quarter circle
    inside_circle = (x**2 + y**2) <= 1
    
    # π/4 = (area of quarter circle) / (area of square)
    pi_estimate = 4 * np.sum(inside_circle) / n_samples
    
    return pi_estimate, x, y, inside_circle

# Run estimation with increasing samples
sample_sizes = [10, 100, 1000, 10000]
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for idx, n in enumerate(sample_sizes):
    pi_est, x, y, inside = estimate_pi(n)
    row, col = idx // 2, idx % 2
    
    # Plot points
    axes[row, col].scatter(x[inside], y[inside], c='red', s=1, alpha=0.5, label='Inside')
    axes[row, col].scatter(x[~inside], y[~inside], c='blue', s=1, alpha=0.5, label='Outside')
    
    # Draw quarter circle
    theta = np.linspace(0, np.pi/2, 100)
    axes[row, col].plot(np.cos(theta), np.sin(theta), 'g-', linewidth=2)
    
    axes[row, col].set_xlim(0, 1)
    axes[row, col].set_ylim(0, 1)
    axes[row, col].set_aspect('equal')
    axes[row, col].set_title(f'n={n}: π ≈ {pi_est:.4f} (error: {abs(pi_est - np.pi):.4f})')
    axes[row, col].legend()
    axes[row, col].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Example 2: Convergence of Monte Carlo estimate
n_trials = 100
max_samples = 10000
sample_points = np.linspace(100, max_samples, 50, dtype=int)

estimates = []
errors = []

for n in sample_points:
    # Run multiple trials
    trial_estimates = [estimate_pi(n)[0] for _ in range(n_trials)]
    mean_estimate = np.mean(trial_estimates)
    estimates.append(mean_estimate)
    errors.append(np.std(trial_estimates))

estimates = np.array(estimates)
errors = np.array(errors)

# Plot convergence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(sample_points, estimates, 'b-', linewidth=2, label='MC Estimate')
axes[0].fill_between(sample_points, estimates - errors, estimates + errors, 
                      alpha=0.3, label='±1 Std Dev')
axes[0].axhline(np.pi, color='red', linestyle='--', linewidth=2, label='True π')
axes[0].set_xlabel('Number of Samples')
axes[0].set_ylabel('Estimate of π')
axes[0].set_title('Monte Carlo Convergence')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot error vs sample size (should decrease as 1/sqrt(n))
abs_errors = np.abs(estimates - np.pi)
theoretical_error = 1 / np.sqrt(sample_points)

axes[1].loglog(sample_points, abs_errors, 'b-', linewidth=2, label='Observed Error')
axes[1].loglog(sample_points, theoretical_error * 2, 'r--', linewidth=2, 
               label='1/√n (theoretical)')
axes[1].set_xlabel('Number of Samples')
axes[1].set_ylabel('Absolute Error')
axes[1].set_title('Error Decreases as 1/√n')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"True π: {np.pi:.6f}")
print(f"Final MC estimate: {estimates[-1]:.6f}")
print(f"Error: {abs(estimates[-1] - np.pi):.6f}")
```

## Chapter 10: Confidence Intervals and Hypothesis Testing (Making Decisions Under Uncertainty)

**What It Is:** How confident should we be in our conclusions from data? Statistical inference gives us tools to quantify uncertainty.

### Confidence Intervals

A confidence interval gives a range of plausible values for a parameter.

**95% Confidence Interval for the mean:**
$\bar{x} \pm 1.96 \times \frac{s}{\sqrt{n}}$

**Interpretation:** If we repeated this process many times, 95% of the intervals would contain the true mean.

**WRONG Interpretation:** "There's a 95% probability the true mean is in this interval" (the true mean is fixed, not random!)

### Hypothesis Testing

**The Setup:**
- **Null Hypothesis ($H_0$):** The "boring" hypothesis (e.g., "the drug has no effect")
- **Alternative Hypothesis ($H_1$):** What we're trying to show (e.g., "the drug works")

**The Process:**
1. Assume $H_0$ is true
2. Calculate how surprising our data would be under $H_0$
3. If data is very surprising (p-value < 0.05), reject $H_0$

**p-value:** The probability of seeing data this extreme if $H_0$ is true.

**Why It Matters:**
- **A/B testing:** Is variant B better than A?
- **Feature selection:** Does this feature improve the model?
- **Model comparison:** Is model A significantly better than model B?
- **Causal inference:** Does X cause Y?

**Warning:** p-hacking and multiple testing problems are real! Understand the limitations.

### 💻 Practical Python: Confidence Intervals and t-tests

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# 1. Confidence Interval Demonstration
true_mean = 100
true_std = 15
n_samples = 30
n_experiments = 100

# Run many experiments
intervals = []
contains_true_mean = []

for _ in range(n_experiments):
    sample = np.random.normal(true_mean, true_std, n_samples)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    
    # 95% confidence interval using t-distribution
    margin = stats.t.ppf(0.975, n_samples-1) * sample_std / np.sqrt(n_samples)
    ci_lower = sample_mean - margin
    ci_upper = sample_mean + margin
    
    intervals.append((ci_lower, ci_upper))
    contains_true_mean.append(ci_lower <= true_mean <= ci_upper)

# Plot confidence intervals
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Show first 50 intervals
for i in range(50):
    color = 'green' if contains_true_mean[i] else 'red'
    axes[0].plot([intervals[i][0], intervals[i][1]], [i, i], color=color, linewidth=2)
    axes[0].scatter([(intervals[i][0] + intervals[i][1])/2], [i], color=color, s=20)

axes[0].axvline(true_mean, color='blue', linestyle='--', linewidth=2, label='True Mean')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Experiment Number')
axes[0].set_title(f'95% Confidence Intervals\n'
                   f'{np.sum(contains_true_mean)}/{n_experiments} contain true mean')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Hypothesis Testing: t-test
# Scenario: Testing if a new drug reduces blood pressure

# Control group (no drug)
control = np.random.normal(120, 10, 50)

# Treatment group (drug reduces by 5 points on average)
treatment = np.random.normal(115, 10, 50)

# Perform two-sample t-test
t_statistic, p_value = stats.ttest_ind(control, treatment)

# Plot distributions
x_range = np.linspace(90, 150, 1000)
axes[1].hist(control, bins=15, density=True, alpha=0.5, color='red', 
             label=f'Control (μ={np.mean(control):.1f})', edgecolor='black')
axes[1].hist(treatment, bins=15, density=True, alpha=0.5, color='green',
             label=f'Treatment (μ={np.mean(treatment):.1f})', edgecolor='black')

# Add vertical lines for means
axes[1].axvline(np.mean(control), color='red', linestyle='--', linewidth=2)
axes[1].axvline(np.mean(treatment), color='green', linestyle='--', linewidth=2)

axes[1].set_xlabel('Blood Pressure')
axes[1].set_ylabel('Density')
axes[1].set_title(f't-test: p-value = {p_value:.4f}\n' +
                   ('Reject H₀: Drug has effect!' if p_value < 0.05 
                    else 'Fail to reject H₀'))
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Confidence Intervals: {np.sum(contains_true_mean)}/{n_experiments} contain true mean")
print(f"\nHypothesis Test:")
print(f"  Control mean: {np.mean(control):.2f}")
print(f"  Treatment mean: {np.mean(treatment):.2f}")
print(f"  t-statistic: {t_statistic:.3f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Conclusion: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'} (α=0.05)")

# 3. Multiple Testing Problem
# Run 100 tests where H₀ is actually true
n_tests = 100
p_values = []

for _ in range(n_tests):
    # Both samples from same distribution (H₀ is true!)
    sample1 = np.random.normal(0, 1, 30)
    sample2 = np.random.normal(0, 1, 30)
    _, p = stats.ttest_ind(sample1, sample2)
    p_values.append(p)

# How many "significant" results?
false_positives = np.sum(np.array(p_values) < 0.05)

plt.figure(figsize=(10, 6))
plt.hist(p_values, bins=20, edgecolor='black', alpha=0.7, color='coral')
plt.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05 threshold')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.title(f'Multiple Testing Problem\n{false_positives}/{n_tests} false positives (expected ~5)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"\nMultiple Testing: {false_positives} false positives out of {n_tests} tests")
print(f"Expected: ~{n_tests * 0.05:.0f} (5% of tests)")
```

## Summary: The Probability Toolkit for ML

1. **Random Variables** are the foundation - every input, output, and parameter is a random variable with uncertainty.

2. **Probability Distributions** describe the shape of uncertainty. Know your distributions: Normal for continuous values, Bernoulli/Binomial for binary outcomes.

3. **Expected Value and Variance** summarize distributions. Your model predicts means and should quantify variance (uncertainty).

4. **Joint and Conditional Probability** model relationships between variables. This is how we encode dependencies in our data.

5. **Bayes' Theorem** is the foundation of learning - it tells us how to update beliefs given evidence. This is the core of Bayesian ML.

6. **Maximum Likelihood Estimation** connects probability to optimization. Minimizing loss = Maximizing probability of data.

7. **Information Theory** (Entropy, Cross-Entropy, KL Divergence) quantifies uncertainty and gives us our loss functions.

8. **Central Limit Theorem** explains why the Normal distribution appears everywhere and justifies many statistical methods.

9. **Monte Carlo Methods** let us approximate anything we can't calculate exactly - essential for modern ML.

10. **Statistical Inference** (Confidence Intervals, Hypothesis Testing) helps us make rigorous decisions despite uncertainty.

## Connections to Machine Learning

**Linear Regression:** MLE with Gaussian noise → minimize squared error

**Logistic Regression:** MLE with Bernoulli distribution → minimize cross-entropy

**Neural Networks:** Backprop + MLE → minimize negative log-likelihood (cross-entropy loss)

**Naive Bayes:** Direct application of Bayes' theorem with independence assumption

**Decision Trees:** Use entropy/information gain to choose splits

**Reinforcement Learning:** Expected rewards, policy gradients, Monte Carlo estimation

**Generative Models:** Model $P(X)$ or $P(X|Y)$ directly using probability distributions

**Uncertainty Quantification:** Bayesian neural networks, MC dropout, ensembles

**A/B Testing:** Hypothesis testing to make business decisions

Understanding probability doesn't just help you use ML tools - it helps you understand *why* they work and *when* they fail.