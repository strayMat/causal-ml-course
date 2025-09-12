###############
Course syllabus
###############

=======
Content
=======

Eight sessions of three hours presenting important concepts and methods on
machine learning for econometrics. The course focuses on flexible models and causal inference in high dimensions. Most of the sessions will display a mix between theoretical considerations and practical application with hands-on in python or R.

The last session is a presentation of the course evaluation project by groups of students.

For now, the website covers only the contents for the following topics:

- 1. Statistical learning and regularized linear models

- 2. Flexible models for tabular data

- 3. Reminders of potential outcomes and Directed Acyclic Graphs

- 4.a. Event studies: Causal methods for panel data

==========
Motivation
==========

High dimension: sparsity in confounders (lasso, double lasso), nonlinearities in confounders (double ML), heterogeneities of effects (generic ML).

=========================================================================
Session 1 -- Statistical learning and regularized linear models
=========================================================================

- Reminder of statistical learning: Bias variance tradeoff, appropriate representation, over/under-fitting

- Regularized regression : lasso, ridge, elastic net, post-lasso

- Practical session:

  - Common pitfalls in the interpretation of coefficients of linear models

  - `Wage analysis with regularized models <https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/python_ml_for_wage_prediction.ipynb>`__

- References:

  - :cite:t:`loic_esteve_2022_7220307`

  - :cite:t:`hastie2017elements`

==============================================
Session 2 -- Flexible models for tabular data
==============================================

- Trees, random forests, boosting

- Cross-validation, nested cross-validation

- Practical session: Hyper-parameters selection for flexible models

- References:

  - :cite:t:`loic_esteve_2022_7220307`

  - :cite:t:`hastie2017elements`

  - :cite:t:`murphy2022probabilistic`

==============================================================================
Session 3 -- Potential outcomes, Directed Acyclic Graphs, confounder selection
==============================================================================

- Reminder on causal inference: prediction/causation, potential outcomes, asking a sound causal question (PICO)

- Causal graph, front-door criteria, and valid adjustment sets.

- Practical Session: DAGs, valid and invalid adjustment sets, with simple linear models and simulations.

- References:

  - :cite:t:`chernozhukov2024applied`, `chapter 2 <https://causalml-book.org/assets/chapters/CausalML_chap_2.pdf>`_ , `chapter 3 <https://causalml-book.org/assets/chapters/CausalML_chap_3.pdf>`_ , `chapter 4 <https://causalml-book.org/assets/chapters/CausalML_chap_4.pdf>`_ , `chapter 5 <https://causalml-book.org/assets/chapters/CausalML_chap_5.pdf>`_ , `chapter 6 <https://causalml-book.org/assets/chapters/CausalML_chap_6.pdf>`_ , `chapter 7 <https://causalml-book.org/assets/chapters/CausalML_chap_7.pdf>`_ , `chapter 8 <https://causalml-book.org/assets/chapters/CausalML_chap_8.pdf>`_

  - :cite:t:`wager2020stats`, Chapter 1

  - :cite:t:`vanderweele2019principles`

===========================================================
Session 4a -- Event studies: Causal methods for panel data
===========================================================

- A causal approach to Difference-in-Differences

- Synthetic controls

- Interrupted time series analysis and state space models

- Practical session: Comparison of different methods for panel data

- References:

 - :cite:t:`chernozhukov2024applied`, `chapter 4 <https://causalml-book.org/assets/chapters/CausalML_chap_4.pdf>`_

 - :cite:t:`gaillac2019machine`, Chapter 8

=====================================================
Session 4b -- Double-lasso for statistical inference
=====================================================

- Partial linear model

- Double-lasso, introduction to Neyman-orthogonality

- Practical session: `Wage analysis from a statistical inference point of view <https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/python_ml_for_wage_prediction.ipynb>`__

- References:

  - :cite:t:`chernozhukov2024applied`, `chapter 4 <https://causalml-book.org/assets/chapters/CausalML_chap_4.pdf>`_

  - :cite:t:`wager2020stats`, Chapter 4

  - :cite:t:`gaillac2019machine`, Chapter 2


==========================================================
Session 5 -- Double machine learning: Neyman-orthogonality
==========================================================

- Importance of sample splitting for double machine learning

- Double-robust estimator approach (also known as augmented inverse propensity weighting)

- Debiased (or double) machine learning, Neyman-orthogonality, method-of-moments

- Practical session: `The Effect of Gun Ownership on Gun-Homicide Rates <https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM4/python_dml_inference_for_gun_ownership.ipynb#scrollTo=hOcTlYfPi-5z>`__

- References:

  - :cite:t:`chernozhukov2024applied`, `chapter 10 <https://causalml-book.org/assets/chapters/CausalML_chap_10.pdf>`_

  - :cite:t:`gaillac2019machine`, Chapter 2

  - :cite:t:`abadie2021using`

=============================================
Session 6 -- Heterogeneous treatment effect
=============================================

- Learners : S, T, X, R learners

- Best linear approximation

- Causal forests

- Practical session: `CATE estimation on 401(k) dataset <https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/T/CATE-estimation.ipynb>`__

- References:

  - :cite:t:`chernozhukov2024applied`, `chapter 14 <https://causalml-book.org/assets/chapters/CausalML_chap_14.pdf>`_

  - :cite:t:`wager2018estimation`

  - :cite:t:`nie2021quasi`

=============================================
Session 7 -- Heterogeneous treatment effect
=============================================

- Practical session:
  - :cite:t:`gaillac2019machine`, Chapter 6

  - :cite:t:`kitagawa2018should`

============
Bibliography
============

.. bibliography:: _static/slides/biblio.bib
   :cited:
