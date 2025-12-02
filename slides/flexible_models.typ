// Use touying for slides
// documentation link : https://typst.app/universe/package/touying

#import "@preview/touying:0.6.1": *
#import "@preview/embiggen:0.0.1": * // LaTeX-like delimiter sizing for Typst
#import "@preview/codly:1.3.0": * // Code highlighting for Typst
#show: codly-init
#import "@preview/codly-languages:0.1.10": * // Code highlighting for Typst
#codly(languages: codly-languages)
#import "@preview/showybox:2.0.4": showybox

#import themes.metropolis: *

#show: metropolis-theme//.with(footer: [ENSAE, Introduction course])

// Make the paper dimensions fit for a presentation and the text larger
//#set page(paper: "4:3")
#set text(size: 22pt)
#set par(justify: true)
#set figure(numbering: none)
//#show math.equation: set text(font: "Fira Math")
#set strong(delta: 100)
#show figure.caption: set text(size: 18pt)

#let c_control(body) = {
  set text(fill: orange)
  body
}


#let argmin(body) = {
  $#math.op("argmin", limits: true) _(body)$
}

#let c_treated(body) = {
  set text(fill: blue)
  body
}

#let eq(body, size: 1.4em) = {
  set align(center)
  set text(size: size)
  body
}

// Slides are provided by touying/metropolis; no custom slide macro needed


#let my_box(title, color, body) = {
  showybox(
    title-style: (
      color: black,
      weight: "regular",
      boxed-style: (
        anchor: (
          x: left,
          y: horizon,
        ),
        radius: 10pt,
      ),
    ),
    frame: (
      border-color: color.darken(50%),
      title-color: color.lighten(60%),
      body-color: color.lighten(80%),
    ),
    title: title,
    align(center, body),
  )
}

#let hyp_box(title: "Assumption", body) = {
  my_box(title, rgb("#d39e57"), body)
}


#let def_box(title: "Definition", body) = {
  my_box(title, rgb("#57b4d3"), body)
}

#show link: set text(fill: rgb("#3788d3"))


#title-slide(
  author: [Matthieu Doutreligne],
  title: "Machine Learning for econometrics",
  subtitle: "Flexible models for tabular data",
  date: "February 18th, 2025",
  extra: [A lot of today's content is taken from the excellent #link("https://inria.github.io/scikit-learn-mooc/toc.html", "sklearn mooc") @loic_esteve_2022_7220307],
)

#slide(title: "Reminder from previous session")[

  - Statistical learning 101: bias-variance trade-off

  - Regularization for linear models: Lasso, Ridge, Elastic Net

  - Transformation of variables: polynomial regression

  #uncover(2)[🤔 But... How to select the best model? the best hyper-parameters?]
]


#slide(title: "Table of contents")[
  #outline(depth: 1)
]

#new-section-slide("Model evaluation and selection with cross-validation")

#slide(title: "A closer look at model evaluation: Wage example")[
  == Example with the Wage dataset

  - Raw dataset: (N=534, p=11)

  #only(1)[
    #figure(image("img/flexible_models/wage_head.png"))
  ]

  #uncover((2, 3))[
    - Transformation: encoding categorical data, scaling numerical data: (N=534, p=23)
  ]

  #only(2)[
    #figure(image("img/flexible_models/wage_transformed_head.png"))
  ]

  #uncover(3)[
    - Regressor: Lasso with regularization parameter ($alpha=10$)
  ]
  #only(3)[
    #figure(
      image(
        "img/flexible_models/wage_pipeline.png",
        width: 50%,
      ),
    )
  ]
]

#slide(title: "Repeated train/test splits")[
  #only(1)[
    == Splitting once: In red, the training set, in blue, the test set

    #figure(
      image(
        "img/pyfigures/train_test_split_visualization_seed_0.png",
        width: 65%,
      ),
    )
  ]
  #only(2)[
    == But we could have chosen another split ! Yielding a different MAE

    #figure(
      image(
        "img/pyfigures/train_test_split_visualization_seed_1.png",
        width: 65%,
      ),
    )
  ]

  #only(3)[
    == And another split...

    #figure(
      image(
        "img/pyfigures/train_test_split_visualization_seed_3.png",
        width: 65%,
      ),
    )
  ]

  #only(4)[
    == Splitting ten times

    #figure(
      image(
        "img/pyfigures/train_test_split_visualization_seed_9.png",
        width: 65%,
      ),
    )
    == 🎉 Distribution of MAE: $3.71 plus.minus 0.26$
  ]
]

#slide(title: "Repeated exclusive train/test splits = Cross-validation")[

  Practical usage with sklearn: `cross_validate`.

  #only((1, 2))[
    ```python
    from sklearn.model_selection import cross_validate
    cv_results = cross_validate(
        regressor, data, target, cv=5, scoring="neg_mean_absolute_error"
    )
    ```
  ]
  #only(2)[
    - 🙂 Robustly estimate generalization performance.
    - 🤩 Estimate data variability of the performance : bigger source of variation @bouthillier2021accounting.
    - 🚀 Let's use it to select the best models among several candidates!
  ]
]

// #slide(title: "Repeated non-exclusive train/test splits = Bootstrapping")[


//   #only(1)[
//     === Practical usage with sklearn: `cross_validate` and `ShuffleSplit`.
//     ```python
//     from sklearn.model_selection import cross_validate
//     from sklearn.model_selection import ShuffleSplit

//     cv = ShuffleSplit(n_splits=40, test_size=0.3, random_state=0)
//     cv_results = cross_validate(
//         regressor, data, target, cv=cv, scoring="neg_mean_absolute_error"
//     )
//     ```
//   ]
//   #only(2)[
//     === Why?
//     - ⚠️ Cross-validation underestimates the variability of the performance @bates2024cross
//    - 🚀 Nested cross-validation helps increasing nominal coverage of the confidence intervals.
//   ]
// ]

#slide(title: [Cross-validation for model selection: choose best $alpha$ for lasso])[
  === Wage pipeline

  #only(1)[
    #figure(
      image(
        "img/flexible_models/wage_pipeline.png",
        width: 70%,
      ),
    )
  ]

  #only((2, 3))[
    === Random search over a distribution of $alpha$ values
  ]

  #only(2)[
    #set text(size: 20pt)
    ```python
    param_distributions = {"lasso__alpha": loguniform(1e-6, 1e3)}
    model_random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=10, # number of hyper-parameters sampled
        cv=5, # number of folds for the cross-validation
        scoring="neg_mean_absolute_error", # score to optimize
    )
    model_random_search.fit(X, y)
    ```
  ]

  #only(3)[
    === Goal: Identify the best $alpha$ value(s)
    #figure(
      image(
        "img/pyfigures/lasso_random_search_cv.svg",
        width: 50%,
      ),
    )
  ]
]


#slide(title: "What final model to use for new prediction?")[
  - Often used in practice: #alert[refit on the full data] the model with the best hyper-parameters.

  #uncover((2, 3))[
    - Theorically motivated: Aggregate the outputs from the cross-validate estimators of the best model:
    #eq[$hat(y) = 1/K sum_(k=1)^K hat(y)_k$] #h(2em)where $hat(y)_k$ is the prediction of the model trained on the $k$-th fold.
  ]

  #uncover(3)[
    - #text(fill: orange)[Averaging cross-validate estimators selects the best model] asymptotically among a family of models @lecue2012oracle
  ]
]

#slide(title: [Naive cross-validation to select and estimate the best performances])[
  == Hyper-parameters selection is a kind of model fitting

  Using a single loop of cross-validation, the full dataset is used to:
  - #alert[Select] the best hyper-parameters
  - #alert[Estimate] the generalization performance of the selected model

  #uncover((2, 3))[
    == ⚠️ Naive cross-validation can lead to overfitting and over-optimistic performance estimation
  ]
  #uncover(3)[
    == 🚀 Solution: Nested cross-validation @varoquaux2017assessing
  ]
]

#slide(title: [Nested cross-validation to select and estimate the best performances])[
  - Inner CV loop to select the best hyper-parameters
  - Outer loop to estimate the generalization performance of the selected model
  #figure(image("img/flexible_models/nested_cross_validation.png", width: 70%))
]


#slide(title: "Over-optimistic performance estimation: example")[
  #grid(columns: (1fr, 1fr), gutter: 3mm)[
    - Dataset: Breast cancer (N, p) = (569, 30)
    - Classifier: RandomForestClassifier with multiple choices of hyper-parameter
  ][
    #figure(
      image(
        "img/pyfigures/3_flexible_models_cross_validation_nested_overfitting.svg",
        width: 100%,
      ),
    )
  ]
]


#new-section-slide("Flexible models: Tree, random forests and boosting")

#slide(title: "What is a decision tree? An example.")[
  #figure(image("img/flexible_models/tree_example.svg", width: 80%))
]

#slide(title: "Growing a classification tree")[
  #grid(columns: (1fr, 1fr), gutter: 3mm)[
    #only(1)[
      #figure(image("img/flexible_models/tree_blue_orange1.svg", width: 90%))
    ]
    #only(2)[
      #figure(image("img/flexible_models/tree_blue_orange2.svg", width: 90%))
    ]
    #only(3)[
      #figure(image("img/flexible_models/tree_blue_orange3.svg", width: 90%))
    ]
  ][
    #only(1)[
      #figure(image("img/flexible_models/tree2D_1split.svg", width: 90%))
    ]
    #only(2)[
      #figure(image("img/flexible_models/tree2D_2split.svg", width: 90%))
    ]
    #only(3)[
      #figure(image("img/flexible_models/tree2D_3split.svg", width: 90%))
    ]
  ]
]


#slide(title: "Growing a regression tree")[
  #grid(columns: (1fr, 1fr), gutter: 3mm)[
    #only(1)[
      #figure(image("img/flexible_models/tree_regression_structure1.svg", width: 90%))
    ]
    #only(2)[
      #figure(image("img/flexible_models/tree_regression_structure2.svg", width: 90%))
    ]
    #only(3)[
      #figure(image("img/flexible_models/tree_regression_structure3.svg", width: 90%))
    ]
  ][
    #only(1)[
      #figure(image("img/flexible_models/tree_regression2.svg", width: 90%))
    ]
    #only(2)[
      #figure(image("img/flexible_models/tree_regression3.svg", width: 90%))
    ]
    #only(3)[
      #figure(image("img/flexible_models/tree_regression4.svg", width: 90%))
    ]
  ]
]


#slide(title: "How the best split is chosen?")[
  === The best split minimizes an impurity criteria
  - For the next left and right nodes
  - Over all features
  - And all possible splits

  #pause
  === Formally

  Let the data at node $m$ be $Q_m$ with $n_m$ samples. For a candidate split on feature $j$ and threshold $t_m$ $theta=(j, t_m)$, the split yields: \

  #align(center)[$Q_m^("left")(theta)={(x,y)|x_j<=t_m}$ and $Q_m^("right")(theta)=Q_m backslash Q_m^("left")(theta)$]

  #pause

  Then $theta^(*)$ is chosen to minimize the impurity criteria averaged over the two children nodes:

  $theta^(*) = "argmin"_(j, t_m) [n_m^"left" / n_m H(Q_m^("left")(theta)) + n_m^"right" / n_m H(Q_m^("right")(theta))]$ with $H$ the impurity criteria.
]

#slide(title: "Impurity criteria")[

  == Classification
  === Gini impurity
  #align(center)[
    $H(Q_m) = sum_k p_(m k) (1 - p_(m k))$ with $p_(m k) = 1 / n_m sum_(y in Q_m) I(y = k)$]
  #pause

  === Cross-entropy
  #align(center)[
    $H(Q_m) = - sum_(k in K) p_(m k) log(p_(m k))$
  ]
  #pause

  == Regression
  === Mean squared error

  #align(center)[$H(Q_m) = 1 / n_m sum_(y in Q_m) (y - dash(y_m))^2$ where $dash(y_m) = 1 / n_m sum_(y in Q_m) y$
  ]
]

#slide(title: "Chose the best split: example")[
  #only(1)[
    #grid(columns: (1fr, 2fr), gutter: 3mm)[
      == Random split
    ][
      figure(image("img/pyfigures/tree_random_split.svg", width: 110%))
    ]
  ]
  #only(2)[
    #grid(columns: (1fr, 2fr), gutter: 3mm)[
      == Moving the split to the right from one point
    ][
      figure(image("img/pyfigures/tree_split_2.svg", width: 110%))
    ]

  ]
  #only(3)[
    #grid(columns: (1fr, 2fr), gutter: 3mm)[
      == Moving the split to the right from 10 points
    ][
      figure(image("img/pyfigures/tree_split_10.svg", width: 110%))
    ]
  ]
  #only(4)[
    #grid(columns: (1fr, 2fr), gutter: 3mm)[
      == Moving the split to the right from 20 points
    ][
      figure(image("img/pyfigures/tree_split_19.svg", width: 110%))
    ]
  ]
  #only(5)[
    #grid(columns: (1fr, 2fr), gutter: 3mm)[
      == Best split
    ][
      figure(image("img/pyfigures/tree_best_split.svg", width: 110%))
    ]
  ]
]

#slide(title: "Tree depth and overfitting")[
  #set align(center + top)
  #grid(
    columns: (auto, auto, auto),
    gutter: 1pt,
    [

      #image("img/flexible_models/dt_underfit.svg", width: 90%)
      Underfitting\
      `max_depth` or\
      `max_leaf_nodes` \
      too small],
    [
      #uncover(3)[
        #image("img/flexible_models/dt_fit.svg", width: 90%)
        Best trade-off]
    ],
    [
      #uncover((2, 3))[
        #image("img/flexible_models/dt_overfit.svg", width: 90%)
        Overfitting\
        `max_depth` or\
        `max_leaf_nodes`\
        too large
      ]],
  )
]

#slide(title: "Main hyper-parameters of tree models")[
  #set text(size: 20pt)
  ```python
  DecisionTreeRegressor(
      criterion="squared error",
      max_depth=None, # Tree depth (assume symetric trees)
      min_samples_split=2, # Tree depth (allowing asymetric trees)
      min_samples_leaf=1, # Tree depth (allowing asymetric trees)
      max_leaf_nodes=None, # Tree depth (allowing asymetric trees)
      min_impurity_decrease=0.0, # Tree depth (allowing asymetric trees)
  )
  ```
]

#slide(title: "Pros and cons of trees")[

  = Pros

  - Easy to interpret
  - Handle mixed types of data: numerical, categorical and missing data
  - Handle interactions
  - Fast to fit

  #pause
  = Cons

  - Prone to overfitting
  - Unstable: small changes in the data can lead to very different trees
  - Mostly useful as a building block for ensemble models: random forests and boosting trees
]


#slide(title: "Ensemble models: Bagging ie. Bootstrap AGGregatING")[


  === Bootstrap resampling (random sampling with replacement) proposed by @breiman1996bagging

  === Built upon Bootstrap, introduced by @efron1992bootstrap to estimate the variance of an estimator.

  #pause
  === Bagging is used in machine learning to reduce the variance of a model prone to overfitting

  === Can be used with any model!
]

#slide(title: "Random forests: Bagging with classification trees
")[
  #set align(top)
  #grid(columns: (1fr, 1fr), gutter: 3mm)[
    === Full dataset
    #only((1, 2))[
      #figure(image("img/flexible_models/bagging0.svg", width: 80%))
    ]
    #only(3)[
      #figure(image("img/flexible_models/bagging0_cross.svg", width: 80%))
    ]
  ][
    === Three bootstrap samples
    #only(1)[
      #figure(image("img/flexible_models/bagging.svg", width: 90%))
    ]
    #only(2)[
      #figure(image("img/flexible_models/bagging_line.svg", width: 90%))
      #figure(image("img/flexible_models/bagging_trees.svg", width: 90%))
    ]
    #only(3)[
      #figure(image("img/flexible_models/bagging_cross.svg", width: 83%))
      #figure(image("img/flexible_models/bagging_trees_predict.svg", width: 83%))
      #figure(image("img/flexible_models/bagging_vote.svg", width: 83%))
    ]
  ]
]
#slide(title: "Random forests: Bagging with regression trees")[
  #only(1)[
    #figure(image("img/flexible_models/bagging_reg_data.svg", width: 50%))
  ]
]
#slide(title: "Random forests: Bagging with regression trees")[
  #set align(top)
  #only(1)[
    #figure(image("img/flexible_models/bagging_reg_grey.svg", width: 100%))
  ]
  #only((2, 3))[
    #figure(image("img/flexible_models/bagging_reg_grey_fitted.svg", width: 100%))
  ]
  #h(1em)
  #grid(columns: (1fr, 1fr), gutter: 3mm)[
    #uncover((1, 2, 3))[
      === Bootstrap multiple subsets
    ]
    #uncover((2, 3))[
      === Fit one model to each subset
    ]
    #uncover(3)[
      === Average the predictions
    ]
  ][
    #only(3)[
      #figure(image("img/flexible_models/bagging_reg_blue.svg", width: 60%))
    ]
  ]
]

#slide(title: "Main hyper-parameters of random forests")[
  #set text(size: 18pt)
  ```python
  sklearn.ensemble.RandomForestRegressor(
    n_estimators=100, # Number of trees to fit (sample randomization): not useful to tune in practice
    criterion='squared_error',
    max_depth=None, # tree regularization
    min_samples_split=2, # tree regularization
    min_samples_leaf=1, # tree regularization
    min_impurity_decrease=0.0, # tree regularization
    n_jobs=None, # Number of jobs to run in parallel
    random_state=None, # Seed for randomization
    max_features=1.0, # Number/ratio of features at each split (feature randomization)
    max_samples = None # Number of sample to draw (with replacement) for each tree
   )
  ```
]
//#slide(title: "Therorical advantages of random forests to reduce overfitting")[ ]

#slide(title: "Random Forests are bagged randomized decision trees")[
  = Random forests
  - For each tree a random subset of samples are selected
  - At each split a random subset of features are selected (more randomization)
  - The best split is taken among the restricted subset
  - Feature randomization decorrelates the prediction errors
  - Uncorrelated errors make bagging work better

  #pause
  = Take away

  - Bagging and random forests fit trees independently
  - Each deep tree overfits individually
  - Averaging the tree predictions reduces overfitting
]

#slide(title: "Boosting: Adaptive boosting")[
  == Boosting use multiple iterative models

  - Use of simple underfitting models: eg. shallow trees

  - Each model corrects the errors of the previous one

  #pause
  == Two examples of boosting

  - Adaptive boosting (AdaBoost): reweight mispredicted samples at each step @friedman2000additive

  - Gradient boosting: predict the negative errors of previous models at each step @friedman2001greedy
]

#slide(title: "Boosting: Adaptive boosting, classification example")[
  #grid(columns: (1fr, 1fr), gutter: 3mm)[
    #only(1)[
      #figure(image("img/flexible_models/boosting0.svg"))
    ]
    #only(2)[
      == First prediction:
      #figure(image("img/flexible_models/boosting1.svg"))
    ]
    #only(3)[
      #figure(image("img/flexible_models/boosting2.svg"))
    ]
    #only(4)[
      #figure(image("img/flexible_models/boosting3.svg"))
    ]
  ][
    #only(2)[
      #figure(image("img/flexible_models/boosting_trees1.svg"))
    ]
    #only(3)[
      #figure(image("img/flexible_models/boosting_trees2.svg"))
    ]
    #only(4)[
      #figure(image("img/flexible_models/boosting_trees3.svg"))
      == At each step, AdaBoost weights mispredicted samples
    ]
  ]
]


#slide(title: "Adaboost for classification: choice of the weight")[

  === 🔎 Motivation in @murphy2022probabilistic

  + Initialize the observation weights $w_i = 1/N, i = 1..N$
  #uncover((2, 3, 4, 5, 6, 7, 8))[

    + For m = 1 to M (iterate):
  ]
  #uncover((3, 4, 5, 6, 7, 8))[

    🔸Fit a classifier $F_m (x)$ to the training data using weights $w_i$\
  ]
  #uncover((4, 5, 6, 7, 8))[

    🔸 Compute $"err"_m = (sum_(i=1)^N w_i bb(1)[y_i != F_m (x_i)]) / (sum_(i=1) w_i)$\
  ]
  #uncover((5, 6, 7, 8))[

    🔸 Compute $alpha_m = log((1 - "err"_m )/"err"_m)$\
  ]
  #uncover((6, 7, 8))[

    🔸 Set $w_i arrow w_i exp[alpha_m bb(1)[y_i !=F_m (x_i )], i = 1..N$\
  ]
  #uncover((7, 8))[

    🔸 Output $F(x) = "sign"(sum_(i=1)^M alpha_m G_m(x))$\
  ]
]

#slide(title: "Adaboost: Take-away")[

  - Sequentially fit weak learners (eg. shallow trees)
  - Each new learner corrects the errors of the previous one thanks to sample weights
  - The final model is a weighted sum of the weak learners
  - The weights are learned by the algorithm to given more importance to errors
  - Any weak learner can be used

  #only(2)[
    == Adaboost is tailored to a specific loss function (exponential loss)

    🤔 Can we exploit the boosting idea for any loss function?
  ]
]

#slide(title: "Gradient boosting: how to choose the iterative learners?")[
  === Boosting formulation

  $F_m(x) = F_(m-1)(x)+h_m(x)$ with $F_(m-1)$ the previous estimator, $h_m$, new week learner.

  === Minimization problem

  $h_m = argmin(h) (L_m)=argmin(h) sum_(i=1)^n l(y_i, F_(m-1)(x_i)+h(x_i))$

  Expand the loss inside the sum using #text(fill: orange)[a Taylor expansion.]\

  #uncover((3, 4, 5))[
    $l(y_i, F_(m-1)(x_i) + h(x_i)) = underbrace(l(y_i, F_(m-1)(x_i)), "constant in "h(x_i)) + h(x_i) underbrace((partial l (y_i, F(x_i))) / (partial F(x_i))]_(F=F_(m-1)), \u{225D} g_i", the gradient")$
  ]

  #only((2, 3))[
    #def_box(title: "🔔Taylor expansion")[
      For $l(dot)$ differentiable: $l(y+h) approx l(y) + h (partial l) / (partial y) (y)$
    ]
  ]

  #only(5)[
    Finally:
    $h_m = argmin(h) sum^n_(i=1) h(x_i) g_i$ -> kind of an inner product $<g, h>$

    So $h_m(x_i)$ should be proportional to $- g_i$, so #text(fill: orange)[fit $h_m$ to the negative gradient.]
  ]
]


#slide(title: "Boosting: Gradient boosting, regression example")[
  #grid(columns: (1fr, 1fr), gutter: 3mm)[
    == Regression
    - The loss is: $l(y, F(x)) = (y - F(x))^2$
    - The gradient is: $g_i = -2(y_i - F_(m-1)(x_i))$
    💡The new tree should fit the residuals
  ][
    #figure(image("img/pyfigures/gradient_boosting_data.svg", width: 110%))
  ]
]

#slide(title: "Boosting: Gradient boosting, regression example")[

  #grid(columns: (1fr, 2fr), gutter: 3mm)[
    == Fit a shallow tree (depth=3)
  ][
    #figure(image("img/pyfigures/gradient_boosting_fit.svg", width: 85%))
  ]
]
#slide(title: "Boosting: Gradient boosting, regression example")[

  #grid(columns: (1fr, 2fr), gutter: 3mm)[
    == Fit a second tree to the residuals
    - This tree performs poorly on some samples.
  ][
    #figure(image("img/pyfigures/gradient_boosting_residuals.svg", width: 85%))
  ]
]
#slide(title: "Boosting: Gradient boosting, regression example")[
  #grid(columns: (1fr, 2fr), gutter: 3mm)[
    == Fit a second tree to the residuals
    - This tree performs well on some residuals.
    - Let's zoom on one of those.
  ][
    #figure(image("img/pyfigures/gradient_boosting_residuals_before_zoom.svg", width: 85%))
  ]
]

#slide(title: "Boosting: Gradient boosting, regression example")[

  #grid(columns: (1fr, 2fr), gutter: 3mm)[

    === Focus on a sample
    $(x_i, y_i) = (-0.454, -0.417)$

    === First tree prediction

    Prediction: $f_1(x_i) = -0.016$

    Residuals: #linebreak() $y_i-f_1(x_i)= -0.401$

    #uncover(2)[
      === Second tree prediction
      Prediction: $f_2(x_i) = -0.401$

      Residuals: #linebreak() $y_i-f_1(x_i) - f_2(x_i)= 0$

    ]
  ][
    #only(1)[
      #figure(image("img/pyfigures/gradient_boosting_fit_zoom.svg", width: 85%))
    ]
    #only(2)[
      #figure(image("img/pyfigures/gradient_boosting_residuals_zoom.svg", width: 85%))
    ]
  ]
]


#slide(title: "Faster gradient boosting with binned features")[
  === 😭 Gradient boosting is slow when N>10,000
  Fitting each tree is quite slow: $O(p N log(N))$ operations

  #pause
  === 🚀 XGBoost: eXtreme Gradient Boosting @chen2016xgboost

  - Missing values support
  - Parallelization
  - Second order Taylor expansion

  #pause
  === 🚀 HistGradientBoosting: sklearn implementation of lightGBM @ke2017lightgbm
  - Missing values support
  - Parallelization
  - Discretize numerical features into 256 bins: less costly for tree splitting

]

#slide(title: "Take away for ensemble models")[

  #table(
    columns: 2,
    table.header([#text(fill: orange)[Bagging] (eg. *Random forests*)], [#text(fill: orange)[Boosting]]),

    "Fit trees independently", "Fit trees sequentially",
    "Each deep tree overfits", "Each shallow tree underfits",
    "Averaging the tree predictions reduces overfitting", "Sequentially adding trees reduces underfitting",
  )

]


#new-section-slide("A word on other families of models")


#slide(title: "Other well known families of models")[

  == Generalized linear models
  Link an OLS ($X beta$) to the parameters of various probability distributions.\
  Examples: Poisson regression (for count data), logistic regression.

  #pause
  == Kernels: support vector machines, gaussian processes
  Local methods with appropriate basis functions (kernels).\
  Kernels are often chosen with expert knowledge.

  #pause
  == Deep neural networks (deep learning)
  Iterative layers of parametrized basis functions: eg. $bb(1)[w X + b >= 0]$\
  Trainable by gradient descent: each layer should be differentiable.\
  Training thanks to backpropagation ie. automatic differentiation and gradient methods.
]


#slide(title: "A word on deep learning")[

  #grid(columns: (1fr, 1fr), gutter: 3mm)[
    === Success of deep learning
    🔸 For images: Convolutional Neural Network (CNN) architecture @russakovsky2015imagenet,\
    🔸 For text: transformer architecture @vaswani2017attention,\
    🔸 For protein folding: transformer architecture @jumper2021highly\

    #uncover(2)[
      🤔 Why not so used in econometrics?
    ]
  ][
    #figure(
      image("img/flexible_models/imagenet.png", width: 90%),
      caption: [Imagenet challenge @russakovsky2015imagenet],
    )

  ]

]


#slide(title: [Answer 1: Limited data settings (typically $N approx 1$ million)])[
  - Typically #only(1)[in economics] (but also in industry), we have a limited number of observations

  #figure(
    image("img/flexible_models/2020_kdd_dataset_sizes.png", width: 65%),
    caption: [Typical dataset are mid-sized. This does not change with time. #footnote("https://www.kdnuggets.com/2020/07/poll-largest-dataset-analyzed-results.html")],
  )
]

#slide(title: "Answer 2: Deep learning underperforms on data tables")[

  === Tailored deep learning architectures lack appropriate prior of tabular data @grinsztajn2022tree

  #figure(image("img/flexible_models/tree_outperforms_dl.png", width: 83%))
]

#slide(title: "Nuance: recent work using deep learning for tabular data")[
  #grid(columns: (1fr, 1fr), gutter: 3mm)[
    === Learning appropriate representations (prior) of tabular data

    - TabPFN: large scale pretraining a transformer based model on synthetic tabular data @hollmann2025accurate

    - Allows In-Context Learning (ICL): learn with few examples.

    #uncover(2)[
      - @hollmann2025accurate Figure 4a:\ Comparison on test benchmarks, 29 classification and 28 regression datasets, containing with up to 10,000 samples and 500 features.
    ]
  ][
    #figure(image("img/flexible_models/tabpfn.png", width: 44%))
  ]
]

#slide(title: "Nuance: recent work using deep learning for tabular data")[
  #grid(columns: (1fr, 1fr), gutter: 3mm)[
    == Using Large Language Models (LLM)
    - #text(fill: oklab(45.46%, -0.026, -0.31))[Tabula 8B] Fine-tuning existing LLM (Llama 3-8B) on tabular data @gardner2024large

    #uncover(2)[
      - Allow ICL with few examples.
      - But requires large computational resources and is outperform rapidly when number of samples grows.
    ]
  ][
    #figure(image("img/flexible_models/tabula8B.png", width: 80%))
  ]
]


#slide(title: "Nuance: recent work using deep learning for tabular data")[
  #grid(columns: (1fr, 1fr), gutter: 3mm)[
    === Transferable components tailored to tabular data

    - CARTE: tailored learning components such as `{key:value}` representations  @kim2024carte
    - TABICL: Combine tailored components and pretraining on synthetic data @qu2025tabicl
    - @qu2025tabicl Figure 5:\ Benchmark accuracy results and train times on 200 classification datasets.
  ][
    #figure(image("img/flexible_models/tabicl.png", width: 80%))
  ]
]

#new-section-slide("Python hands-on")

#slide(title: [To your notebooks 🧑‍💻!])[
  - url: https://github.com/strayMat/causal-ml-course/tree/main/notebooks
]


#let bibliography = bibliography("biblio.bib", style: "apa")

#set align(left)
#set text(size: 20pt, style: "italic")
#slide[
  #bibliography
]
