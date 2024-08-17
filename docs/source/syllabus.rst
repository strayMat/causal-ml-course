###############
Course syllabus
###############

=======
Content
=======

Four sessions of three hours presenting important concepts and methods on machine learning for econometrics focusing on modern causal inference. 

Four steps for sound causal inference, illustrated with well known examples from the econometrics literature. 

Common thread : TODO 

==========
Evaluation 
==========

- Articles implementation/read and presentation
- Mini-project (Choosing a running example among different data-sets/problems).

=========
Data sets
=========

[Aller explorer](https://www.data.gouv.fr/fr/pages/donnees_apprentissage-automatique/)
- [Offres d'emplois diffusées à Pôle emploi](https://www.data.gouv.fr/fr/datasets/offres-demploi-diffusees-a-pole-emploi/) , effet sur l'emploi ? 
- Wages and sex ? Un classique expliqué dans le applied causal inference, p.156, que l'on peut refaire jouer avec [la base tout salarié de l'Insee](https://www.insee.fr/fr/statistiques/7651654#dictionnaire).
- [Proposition d'admission](https://www.data.gouv.fr/fr/datasets/parcoursup-propositions-dadmission-dans-lenseignement-superieur-des-eleves-de-terminale-diplomes-du-baccalaureat-general-selon-leurs-enseignements-de-specialite/)
- [The Welfare experiment](https://gssdataexplorer.norc.org/variables/vfilter)
- [Education: Taille des classes sur la réussite au brevet](https://www.data.gouv.fr/fr/pages/donnees_education/)

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

- ATE

- Outcome based identification, propensity score estimation

- Orthogonal Machine Learning : theoretical advantages for high-dimensional data. 

- Identification proofs 

- Sensitivity analysis 

--------------
Coding session
--------------

----------
References
----------

- :cite:t:`chernozhukov2024applied`, `chapter 10 <https://causalml-book.org/assets/chapters/CausalML_chap_10.pdf>`_ , `chapter 11 <https://causalml-book.org/assets/chapters/CausalML_chap_11.pdf>`_ , 

- :cite:t:`wager2020stats`, Lecture 2 and 3

=========
Session 3
=========

- Motivation for Heterogeneous Treatment Effect, optimal policy.

- Model selection for CATE : R loss function

- Causal forest, targeted learning.

--------------
Coding session
--------------

Use example from session 2, but focus on CATE and policy learning.

----------
References
----------

- :cite:t:`chernozhukov2024applied`, `chapter 14 <https://causalml-book.org/assets/chapters/CausalML_chap_14.pdf>`_ , `chapter 15 <https://causalml-book.org/assets/chapters/CausalML_chap_15.pdf>`_ ,

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

- IV (see before ?)


============
Bibliography
============

.. bibliography:: _static/biblio.bib
   :cited:
 
