###############
Course syllabus
###############

=======
Content
=======

Eight sessions of three hours presenting important concepts and methods on
machine learning for econometrics. The course focuses on flexible models and causal inference in high dimensions. Most of the sessions will display a mix between theoretical considerations and practical application with hands-on in python or R.  

Motivation : 

- high dimensions ie. sparsity in confounders (double lasso)
- high dimensions, nonlinearities in confounders (double ML)
- heterogeneities of effects (generic ML) -> only in RCT / optimal assignment



========================================================================
Session 1 -- Directed acyclic graph, valid adjustment sets
========================================================================

- Reminder on causal inference: prediction/causation, potential outcomes, asking a sound causal question (PICO)

- Causal graph, front-door criteria and valid adjustment sets.
 
- Coding session : 
  - Valid and unvalid adjustment sets, with simple linear models and simulations. 
  - `DAGs: D-Separation and Conditonal Independencies, Adjustment via Backdoor and Swigs, Equivalence Classes, Falsifiability Tests. <https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/CM3/python-pgmpy.ipynb>`__

----------
References
----------

- :cite:t:`chernozhukov2024applied`, `chapter 2 <https://causalml-book.org/assets/chapters/CausalML_chap_2.pdf>`_ , `chapter 3 <https://causalml-book.org/assets/chapters/CausalML_chap_3.pdf>`_ , `chapter 4 <https://causalml-book.org/assets/chapters/CausalML_chap_4.pdf>`_ , `chapter 5 <https://causalml-book.org/assets/chapters/CausalML_chap_5.pdf>`_ , `chapter 6 <https://causalml-book.org/assets/chapters/CausalML_chap_6.pdf>`_ , `chapter 7 <https://causalml-book.org/assets/chapters/CausalML_chap_7.pdf>`_ , `chapter 8 <https://causalml-book.org/assets/chapters/CausalML_chap_8.pdf>`_

- :cite:t:`wager2020stats`, Lecture 1

- :cite:t:`vanderweele2019principles`

=========================================================================
Session 2 - reminder of statistical learning, penalized linear regression
=========================================================================

- Reminder of statistical learning: Bias variance tradeoff, appropriate representation, over/under-fitting

- Regularized regression : lasso, ridge, elastic net, post-lasso

- Coding session :

  - `Common pitfalls of interpreting lasso coefficients <https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py>`__

  - `Wage analysis with regularized models <https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/python_ml_for_wage_prediction.ipynb>`__


- :cite:t:`loic_esteve_2022_7220307`

- :cite:t:`hastie2009elements`


============================================
Session 3 - Flexible models for tabular data
============================================

- Trees, random forests, boosting

- Cross-validation, nested cross-validation

- Coding session: 
  
  - Pratical consideration for model selection in high dimension : common metrics, calibration.

  - Wage analysis with flexible models

----------
References
----------

- :cite:t:`loic_esteve_2022_7220307`

- :cite:t:`hastie2009elements`

- :cite:t:`pml1Book`

==================================================
Session 4 : double-lasso for statistical inference
==================================================

- Partial linear model 

- Double-lasso, introduction to Neyman-orthogonality

- Coding session : 

  - `Wage analysis from a statistical inference point of view <https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/python_ml_for_wage_prediction.ipynb>`__

----------
References
----------

- :cite:t:`chernozhukov2024applied`, `chapter 4 <https://causalml-book.org/assets/chapters/CausalML_chap_4.pdf>`_

- :cite:t:`wager2020stats`, lecture 4

- :cite:t:`gaillac2019machine`, lecture 2

====================================
Session 5 -- Neyman-orthogonality
====================================

- 

Coding session: 

  - `The Effect of Gun Ownership on Gun-Homicide Rates <https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM4/python_dml_inference_for_gun_ownership.ipynb#scrollTo=hOcTlYfPi-5z>`__

----------
References
----------

- :cite:t:`chernozhukov2024applied`, `chapter 10 <https://causalml-book.org/assets/chapters/CausalML_chap_10.pdf>`_ 
 


=============================================
Session 6 -- Heterogeneous treatment effect
=============================================




=============================================
Session 7 -- Heterogeneous treatment effect
=============================================

Coding session: 

 -  `Heterogeneous Effect of Sex on Wage Using Double Lasso <https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/python_heterogeneous_wage_effects.ipynb>`__ 




=============================================
Session 8 -- Generic machine learning
=============================================

- 

----------
References
----------

- :cite:t:`chernozhukov2018generic`

==========
Evaluation 
==========

A project on a dataset among those proposed.

---------
Projects 
---------

Run through the different steps of causal inference on a dataset of your choice: asking a sound question, identification, estimation, inference, vibration analysis.

Datasets : 

.. list-table:: Dataset Information
   :header-rows: 1

   * - Dataset Name
     - URL
     - N
     - P
     - Question Example
     - Interventional
   * - Marketing
     - `Link <http://archive.ics.uci.edu/dataset/222/bank+marketing>`__
     - 45211
     - 16
     - "What is the effect of multiple phone call on the term deposit subscription?"
     - No
   * - Nutritional Followup
     - `Link <https://wwwn.cdc.gov/nchs/nhanes/nhefs/>`__
     - Unknown
     - Unknown
     - "How do nutrition habits affect long-term health outcomes?"
     - No
   * - Wages (french version)
     - `Link <https://www.insee.fr/fr/statistiques/7651654#dictionnaire>`__
     - 2403775
     - 31
     - "What factors affect wage disparities?"
     - No
   * - Diabetes 130-US hospitals
     - `Link <http://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>`__
     - 101766
     - 47 
     - "What is the effect of HbA1c measurement on hospital readmission rates at 30 days?"
     - No  
   * - Student's dropout and academic success
     - `Link <http://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>`__
     - 4424
     - 36
     - "What factor influence the dropout of students?"
     - No
   * - Obesity levels in Mexico
     - `Link <http://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition>`__
     - 2111
     - 16
     - "What action is the most effective to prevent obesity?"
     - No

====================================
Other potential sources for Datasets
====================================

- `causal data, mixtape course <https://cran.r-project.org/web/packages/causaldata/causaldata.pdf@>`__

- `Aller explorer <https://www.data.gouv.fr/fr/pages/donnees_apprentissage-automatique/>`__

- `The Welfare experiment <https://gssdataexplorer.norc.org/variables/vfilter>`__

- `UC Irvine ML repository <http://archive.ics.uci.edu/datasets?skip=10&take=10&sort=desc&orderBy=NumHits&search=&NumInstances=572&NumInstances=114237&NumFeatures=12&NumFeatures=3231961>`__


============
Bibliography
============

.. bibliography:: _static/biblio.bib
   :cited:
 
