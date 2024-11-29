###############
Course syllabus
###############

=======
Content
=======

Four sessions of three hours presenting important concepts and methods on
machine learning for econometrics focusing on flexible models and causal inference.

Four steps for sound causal inference, illustrated with well known examples from
the econometrics literature. 

Motivation : 

- high dimensions ie. sparsity in confounders (double lasso)
- high dimensions, nonlinearities in confounders (double ML)
- heterogeneities of effects (generic ML) -> only in RCT / optimal assignment

Important topics: 

- PO reminder, DAG and proper conditionning set, 
- Lasso, and forests,
- Selecting a model for machine learning : statistical learning reminder and scikit-learn best practices 
- Double-lasso for causal inference in PML settings, intro to Neyman-orthogonality.
- Double debiased ML,
- Heterogeneous treatment effect (deux séances) : causal forests, generic ML 
- Optimal assignment, 

Other topics
- importance of calibration when targeting probabilities 
-  (bof),
- Heterogeneous effect : Causal forest? inference on best linear approximation? Meta learning ?  
- vibration analysis 

==========
Evaluation 
==========

A project on a dataset among those proposed.

----------------------------------------
Articles implementation and presentation
----------------------------------------

Articles : 

- [Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. Journal of the Royal Statistical Society Series B: Statistical Methodology, 82(1), 39-67.](https://academic.oup.com/jrsssb/article/82/1/39/7056023) : The recalls and extends the omitted variable bias framework by introducing tools for sensitivity analysis in regression models, accommodating multiple and nonlinear confounders without strong assumptions about their distributions.


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

- [causal data, mixtape course](https://cran.r-project.org/web/packages/causaldata/causaldata.pdf)

- [Aller explorer](https://www.data.gouv.fr/fr/pages/donnees_apprentissage-automatique/)

- [The Welfare experiment](https://gssdataexplorer.norc.org/variables/vfilter)

- [UC Irvine ML repository](http://archive.ics.uci.edu/datasets?skip=10&take=10&sort=desc&orderBy=NumHits&search=&NumInstances=572&NumInstances=114237&NumFeatures=12&NumFeatures=3231961)

=========
Session 1
=========

- Intro to causal inference. Different concepts, similar ideas for applied research from collected data : Intersection of different fields : stats, econometrics/epidemiology,  machine learning. 

- Asking a sound causal question

- Potential outcome notations

- Causal graph, front-door criteria and valid adjustment sets.
 
--------------
Coding session
--------------

- Coding session : Valid and unvalid adjustment sets, with simple linear models and simulations. Asses effects on bias. Take inspiration from [Causal ML, chapter 7 and 8, notebook ](https://colab.research.google.com/github/chernozhukov2024applied/MetricsMLNotebooks/blob/main/CM3/python-pgmpy.ipynb)

----------
References
----------

- :cite:t:`chernozhukov2024applied`, `chapter 2 <https://causalml-book.org/assets/chapters/CausalML_chap_2.pdf>`_ , `chapter 3 <https://causalml-book.org/assets/chapters/CausalML_chap_3.pdf>`_ , `chapter 4 <https://causalml-book.org/assets/chapters/CausalML_chap_4.pdf>`_ , `chapter 5 <https://causalml-book.org/assets/chapters/CausalML_chap_5.pdf>`_ , `chapter 6 <https://causalml-book.org/assets/chapters/CausalML_chap_6.pdf>`_ , `chapter 7 <https://causalml-book.org/assets/chapters/CausalML_chap_7.pdf>`_ , `chapter 8 <https://causalml-book.org/assets/chapters/CausalML_chap_8.pdf>`_

- :cite:t:`wager2020stats`, Lecture 1

- :cite:t:`vanderweele2019principles`

=========
Session 2
=========

- Reminder of statistical learning

- Lasso model 

- Random Forest  

- Boosting ? NN ? 

- Model selection for machine learning, pratical considerations

--------------
Coding session
--------------

[Common pitfalls of interpreting lasso coefficients](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py)  

----------
References
----------

- :cite:t:`loic_esteve_2022_7220307`

- :cite:t:`hastie2009elements`

- :cite:t:`pml1Book`

=========================================================
Session 3 : Lasso for statistical inference, double lasso
=========================================================

- Using 

- Model selection for CATE : R loss function

- Causal forest, targeted learning.

--------------
Coding session
--------------

Use example from session 2, but focus on CATE and policy learning.

----------
References
----------

- :cite:t:`chernozhukov2024applied`, `chapter 4 <https://causalml-book.org/assets/chapters/CausalML_chap_4.pdf>`_

- :cite:t:`wager2020stats`, Lecture 4

====================================
Session 4 -- Methods for time series
====================================

- Difference In Difference

- Synthetic controls

--------------
Coding session
--------------

----------
References
----------

- :cite:t:`chernozhukov2024applied`, `chapter 16 <https://causalml-book.org/assets/chapters/CausalML_chap_16.pdf>`_ 
 
- :cite:t:`abadie2021using`

- :cite:t:`bouttell2018synthetic`

============================
Session 5 -- Advanced topics 
============================

- Going AI : feature engineering and causal inference  

- Proxy causal learning

- IV (seen in another course ?) : is it a good idea to introduce it in a course focused on ML ?


======================================
Reading materials close to the course
=====================================

- [Econometric methods for program evaluation, :cite:t:`abadie2018econometric`](https://www.annualreviews.org/content/journals/10.1146/annurev-economics-080217-053402)



============
Bibliography
============

.. bibliography:: _static/biblio.bib
   :cited:
 
