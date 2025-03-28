==========
Evaluation 
==========

The evaluation is based on a small project on real data making use of some important concepts seen in course. Run through the different steps of causal or predictive inference with machine learning on a dataset of your choice (see below for some propositions of dataset).

You will be evaluated on two deliverables: 

- **Handing over a notebook** detailing both the formalization of your problem, the chosen methods to address the question of interest, and the results of your analysis. The notebook should be clear and well-structured, with a good balance between text and code. You can use either R or python or stata. For causal inference, you can use the `Double ML package <https://docs.doubleml.org/stable/>`__.

- During the last session (April 1st), you will **present your work** to the class. The presentation should be clear and concise, with a focus on the main results and the methodological choices you made. The presentation should be between 20 and 30 min. You can use a slide deck or a live demo of your notebook.

----------
Enrollment
----------

Group: From 4 to 5 students. We will send you by mail a link to register your groups and choice of dataset.

----------------------------
Details plan of the analysis 
----------------------------

The following is a typical plan that you should follow for your analysis. 
The expected steps or concepts are highlighted by a 🧐 emoticon.

- PICO formulation 🧐
- Exploratory data analysis: descriptive analysis, association and visualization 🧐 
- Identification with causal graph and choice of covariates to include 🧐
- Discussion on the assumptions chosen for identification of the effect
- Estimation with an appropriate causal estimator (double post lasso, double machine learning, generic machine learning, inverse propensity score, augmented propensity score) and regressor (lasso, ridge, random forest, boosting). It is not expected that you test or use every method.🧐 
- Parameter/model selection for the statistical estimator (cross-validation) 🧐
- Heterogeneous treatment effects: study the effect of the treatment on different subgroups of the population 
- Sensitivity analysis: placebo check, different models, different covariate sets
- Conclusion and discussion of the results 🧐 

NB: Concluding a valid causal inference study with non-randomized data is not always possible. Crucial confounders could be missing, or other bias could be present in the data production process. The goal of the project is to conduct the most rigorous analysis possible with the data that we are providing you and to discuss the limitations of the study (even if the limits are related to the data). We are not expecting you to be able to robustly conclude on a causal effect in all cases.

---------
Datasets 
---------

Here are some datasets that you can use for the project. You can also use your own dataset but in this case, please add a short justification for the interest of the dataset.

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
     - 45,211
     - 16
     - "What is the effect of multiple phone call on the term deposit subscription?"
     - No
   * - Wages (french version)
     - `Link <https://www.insee.fr/fr/statistiques/7651654#dictionnaire>`__
     - 2,403,775
     - 31
     - "What factors affect wage disparities?"
     - No
   * - Diabetes 130-US hospitals
     - `Link <http://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>`__
     - 101,766
     - 47 
     - "What is the effect of HbA1c measurement on hospital readmission rates at 30 days?"
     - No  
   * - Student's dropout and academic success
     - `Link <http://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>`__
     - 4,424
     - 36
     - "What factor influence the dropout of students?"
     - No
   * - Obesity levels in Mexico
     - `Link <http://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition>`__
     - 2,111
     - 16
     - "What action is the most effective to prevent obesity?"
     - No
   * - Nutritional Followup (hard)
     - `Link <https://github.com/NickCH-K/causaldata/tree/main/Python/causaldata/nhefs>`__ 
     - 1,629
     - 67
     - "How do nutrition habits affect long-term health outcomes?"
     - No


Other potential sources for Datasets:

- `Causal data package: data for the causal mixtape, Huntington-Klein or Hernàn courses <https://github.com/NickCH-K/causaldata/tree/main/Python>`__

----------------------
Criteria for notations 
----------------------

- 20 %: General form of the notebook, clarity of the text, and the structure of the notebook.

- 50 %: Multiple concepts seen during the course are present and appropriately used. Are the results correctly interpreted?

- 15 %: Presence and intelligibility of figures (during the exploration and presenting the results).

- 15 %: Do the analysis and its conclusion make sense? Is it convincing?
