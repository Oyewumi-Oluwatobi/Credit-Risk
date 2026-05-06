# Credit Risk Analytics - Probability of Default & FICO Score Bucketing
**JPMorgan Chase & Co. Quantitative Research Virtual Experience | Forage**

---

## Overview
This project builds an end-to-end credit risk analytics pipeline using a loan dataset of 10,000 customer records. It consists of two main components:

1. **Probability of Default (PD) Model** — predicts the likelihood a customer will default on their loan
2. **FICO Score Bucketing System** — uses dynamic programming to optimally segment customers into risk rating categories

---

## Dataset
- **Source:** JPMorgan Chase Quantitative Research Simulation (Forage)
- **Size:** 10,000 loan records
- **Features:**
  - `credit_lines_outstanding` — number of active credit lines
  - `loan_amt_outstanding` — current outstanding loan amount
  - `total_debt_outstanding` — total debt across all obligations
  - `income` — annual income
  - `years_employed` — years of employment
  - `fico_score` — credit score (408–850)
  - `default` — binary target (1 = defaulted, 0 = did not default)

---

## Part 1: Probability of Default (PD) Model

### Methodology
- **Algorithm:** Logistic Regression (`scikit-learn`)
- **Preprocessing:** StandardScaler normalization applied to all features
- **Train/Test Split:** 80/20 (8,000 training, 2,000 testing records)

### Model Performance
| Metric | Score |
|--------|-------|
| ROC-AUC | **1.0000** |
| Accuracy | **0.9965** |
| Precision | **0.9971** |
| Recall | **0.9828** |

The ROC-AUC score of 1.00 indicates near-perfect discrimination between defaulting and non-defaulting loans.

### Expected Loss (EL) Function
An Expected Loss calculator was built on top of the PD model using the formula:

```
EL = PD × (1 - Recovery Rate) × Loan Amount Outstanding
```

- **Recovery Rate:** 10% (industry standard assumption)
- **Example:** For a loan with $15,000 outstanding, 2 credit lines, $25,000 total debt, $75,000 income, 5 years employed, FICO score 720 → **Calculated EL = $1,073.55**

---

## Part 2: FICO Score Bucketing (Dynamic Programming)

### Problem
Segment continuous FICO scores into discrete risk rating buckets that maximize the predictive power of each bucket for loan default.

### Approach
Used **dynamic programming** to find optimal FICO score boundaries by maximizing the sum of log-likelihoods across all buckets. The log-likelihood for each segment is calculated as:

```
LL = n_i × p_i × log(p_i) + n_i × (1 - p_i) × log(1 - p_i)
```

Where `p_i` is the default rate within segment `i`.

### Results (1,000-record sample, 5 buckets)
| Rating | FICO Range | Avg Default Rate | Record Count |
|--------|-----------|-----------------|--------------|
| Rating 1 | 425–532 | **57.78%** | 45 |
| Rating 2 | 532–607 | **30.30%** | 264 |
| Rating 3 | 607–647 | **17.50%** | 240 |
| Rating 4 | 647–850 | **7.32%** | 451 |

- **Optimal Log-Likelihood:** -411.39
- **Optimal Boundaries:** [532, 607, 647]

The ratings clearly show that higher FICO scores correspond to significantly lower default rates, confirming the bucketing system effectively separates risk levels.

---

## Technologies Used
- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib)
- **Jupyter Notebook** (Google Colab)
- **Algorithms:** Logistic Regression, Dynamic Programming (Log-Likelihood Maximization)

---

## Key Takeaways
- Built a production-ready PD model with near-perfect accuracy on financial loan data
- Developed a custom dynamic programming solution to optimally bin FICO scores by default risk
- Implemented an Expected Loss calculator providing actionable credit risk metrics
- Demonstrated understanding of key financial risk concepts: PD, LGD, EL, FICO scoring
