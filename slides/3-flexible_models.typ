// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.3.1": *
#import "@preview/embiggen:0.0.1": * // LaTeX-like delimiter sizing for Typst
#import "@preview/codly:1.1.1": * // Code highlighting for Typst
#show: codly-init
#import "@preview/codly-languages:0.1.3": * // Code highlighting for Typst
#codly(languages: codly-languages)
#import "@preview/showybox:2.0.1": showybox

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
  $#math.op("argmin", limits: true)_(body)$
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

#let slide(title: none, body) = {
  let header = {
    set align(top)
    if title != none {
      show: m-cell.with(fill: m-dark-teal, inset: 1em)
      set align(horizon)
      set text(fill: m-extra-light-gray, size: 1.2em)
      strong(title)
    } else { [] }
  }

  let footer = {
    set text(size: 0.8em)
    show: pad.with(.5em)
    set align(bottom)
    text(fill: m-dark-teal.lighten(40%), m-footer.display())
    h(1fr)
    text(fill: m-dark-teal, logic.logical-slide.display())
  }

  set page(
    header: header,
    footer: footer,
    margin: (top: 3em, bottom: 1.5em),
    fill: m-extra-light-gray,
  )

  let content = {
    show: align.with(horizon)
    show: pad.with(
      left: 1em,
      top: 0.5em,
      right: 1em,
      bottom: 0em,
    ) // super important to have a proper padding (not 1/3 of the slide blank...)
    set text(fill: m-dark-teal)
    body
  }

  logic.polylux-slide(content)
}


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


#title-slide(
  author: [Matthieu Doutreligne],
  title: "Machine Learning for econometrics",
  subtitle: "Flexible models for tabular data",
  date: "February 18th, 2025",
  //extra: "Extra"
)

#slide(title: "Reminder from previous session")[

  - Statistical learning 101: bias-variance trade-off

  - Regularization for linear models: Lasso, Ridge, Elastic Net

  - Transformation of variables: polynomial regression

  #uncover(2)[- 🤔 But... How to select the best model? the best hyper-parameters?]
]


#slide(title: "Table of contents")[
  #metropolis-outline
]

#new-section-slide("Model evaluation and selection with cross-validation")

#slide(title: "A closer look at model evaluation: Wage example")[
  == Example with the Wage dataset

  - Raw dataset: (N=534, p=11)

  #only(1)[
    #figure(image("img/3-flexible_models/wage_head.png"))
  ]

  #uncover((2, 3))[
    - Transformation: encoding categorical data, scaling numerical data: (N=534, p=23)
  ]

  #only(2)[
    #figure(image("img/3-flexible_models/wage_transformed_head.png"))
  ]

  #uncover(3)[
    - Regressor: Lasso with regularization parameter ($alpha=10$)
  ]
  #only(3)[
    #figure(
      image(
        "img/3-flexible_models/wage_pipeline.png",
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
    == 🎉 Distribution of MAE: $3.71 plus.minus 0.26$ //TODO: find ref for proof (ex. Wager or Causal ML)
  ]
]

#slide(title: "Repeated train/test splits = Cross-validation")[

  = Cross-validation
  - In sklearn, it can be instantiated with `cross_validate`.

  #only(1)[
    ```python
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import ShuffleSplit

    cv = ShuffleSplit(n_splits=40, test_size=0.3, random_state=0)
    cv_results = cross_validate(
        regressor, data, target, cv=cv, scoring="neg_mean_absolute_error"
    )
    ```
  ]
  #only(2)[
    - 🙂 Robustly estimate generalization performance
    - 🤩 Estimate variability of the performance: similar to bootstrapping (but different).
    - 🚀 Let's use it to select the best models among several canditates!
  ]
]

#slide(title: [Cross-validation for model selection: choose best $alpha$ for lasso])[
  - Wage pipeline
  #only(1)[
    #figure(
      image(
        "img/3-flexible_models/wage_pipeline.png",
        width: 70%,
      ),
    )
  ]
  #uncover((2, 3))[
    - Random search over a distribution of $alpha$ values
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
    - Identify the best $alpha$ value(s)
    #figure(
      image(
        "img/pyfigures/lasso_random_search_cv.svg",
        width: 50%,
      ),
    )
  ]
]


#slide(title: "What final model to use for new prediction?")[
  - Either refit on full data the model with the best hyper-parameters on the full data: often used in pratice.

  - Or use the aggregation of outputs from the cross-validation of the best model:
  #eq[$hat(y) = 1/K sum_(k=1)^K hat(y)_k$] #h(2em)where $hat(y)_k$ is the prediction of the model trained on the $k$-th fold.
  - Proof that cross-validation selects the best model asymptotically among a family of models (averaging on the folds): @lecue2012oracle
]

#slide(title: "Naive cross-validation to select AND estimate the best performances")[
  TODO
]

#slide(title: "Nested cross-validation to select the best model")[
  TODO
]

#new-section-slide("Flexible models: Tree, random forests and boosting")

#slide(title: "What is a decision tree? An example.")[
  #figure(image("img/3-flexible_models/tree_example.svg", width: 80%))
]

#slide(title: "Growing a classification tree")[
  #side-by-side(
    [
      #only(1)[
        #figure(image("img/3-flexible_models/tree_blue_orange1.svg", width: 90%))
      ]
      #only(2)[
        #figure(image("img/3-flexible_models/tree_blue_orange2.svg", width: 90%))
      ]
      #only(3)[
        #figure(image("img/3-flexible_models/tree_blue_orange3.svg", width: 90%))
      ]
    ],
    [
      #only(1)[
        #figure(image("img/3-flexible_models/tree2D_1split.svg", width: 90%))
      ]
      #only(2)[
        #figure(image("img/3-flexible_models/tree2D_2split.svg", width: 90%))
      ]
      #only(3)[
        #figure(image("img/3-flexible_models/tree2D_3split.svg", width: 90%))
      ]
    ],
  )
]


#slide(title: "Growing a regression tree")[
  #side-by-side(
    [
      #only(1)[
        #figure(image("img/3-flexible_models/tree_regression_structure1.svg", width: 90%))
      ]
      #only(2)[
        #figure(image("img/3-flexible_models/tree_regression_structure2.svg", width: 90%))
      ]
      #only(3)[
        #figure(image("img/3-flexible_models/tree_regression_structure3.svg", width: 90%))
      ]
    ],
    [
      #only(1)[
        #figure(image("img/3-flexible_models/tree_regression2.svg", width: 90%))
      ]
      #only(2)[
        #figure(image("img/3-flexible_models/tree_regression3.svg", width: 90%))
      ]
      #only(3)[
        #figure(image("img/3-flexible_models/tree_regression4.svg", width: 90%))
      ]
    ],
  )
]


#slide(title: "How the best split is chosen?")[
  === The best split minimizes an impurity criteria
  - for the next left and right nodes
  - over all features
  - and all possible splits

  === Formally

  Let the data at node $m$ be $Q_m$ with $n_m$ samples. For a candidate split on feature $j$ and threshold $t_m$ $theta=(j, t_m)$, the split yields: #linebreak()
  $Q_m^("left")(theta)={(x,y)|x_j<=t_m}$ and $Q_m^("right")(theta)=Q_m backslash Q_m^("left")(theta)$

  Then $theta$ is chosen to minimize the impurity criteria averaged over the two children nodes:

  $theta^(*) = "argmin"_(j, t_m) [n_m^"left" / n_m H(Q_m^("left")(theta)) + n_m^"right" / n_m H(Q_m^("right")(theta))]$ with $H$ the impurity criteria.
]

#slide(title: "Impurity criteria")[

  == For classification
  === Gini impurity

  $H(Q_m) = sum_k p_(m k) (1 - p_(m k))$ with $p_(m k) = 1 / n_m sum_(y in Q_m) I(y = k)$

  === Cross-entropy

  $H(Q_m) = - sum_(k in K) p_(m k) log(p_(m k))$

  == For regression
  === Mean squared error
  $H(Q_m) = 1 / n_m sum_(y in Q_m) (y - dash(y_m))^2$ where $dash(y_m) = 1 / n_m sum_(y in Q_m) y$
]

#slide(title: "Chose the best split: example")[
  #only(1)[
    #side-by-side(
      columns: (1fr, 2fr),
      [== Random split],
      figure(image("img/pyfigures/tree_random_split.svg", width: 110%)),
    )
  ]
  #only(2)[
    #side-by-side(
      columns: (1fr, 2fr),
      [== Moving the split to the right from one point],
      figure(image("img/pyfigures/tree_split_2.svg", width: 110%)),
    )

  ]
  #only(3)[
    #side-by-side(
      columns: (1fr, 2fr),
      [== Moving the split to the right from 10 points],
      figure(image("img/pyfigures/tree_split_10.svg", width: 110%)),
    )
  ]
  #only(4)[
    #side-by-side(
      columns: (1fr, 2fr),
      [== Moving the split to the right from 20 points],
      figure(image("img/pyfigures/tree_split_19.svg", width: 110%)),
    )
  ]
  #only(5)[
    #side-by-side(
      columns: (1fr, 2fr),
      [== Best split],
      figure(image("img/pyfigures/tree_best_split.svg", width: 110%)),
    )
  ]
]

#slide(title: "Tree depth and overfitting")[
  #set align(center + top)
  #grid(
    columns: (auto, auto, auto),
    gutter: 1pt,
    [
      #image("img/3-flexible_models/dt_underfit.svg", width: 90%)
      Underfitting#linebreak()
      max depth or#linebreak()
      max_leaf_nodes#linebreak()
      too small],
    [
      #image("img/3-flexible_models/dt_fit.svg", width: 90%)
      Best trade-off],
    [
      #image("img/3-flexible_models/dt_overfit.svg", width: 90%)
      Overfitting#linebreak()
      max depth or#linebreak()
      max_leaf_nodes#linebreak()
      too large],
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

  = Cons

  - Prone to overfitting
  - Unstable: small changes in the data can lead to very different trees
  - Mostly useful as a building block for ensemble models: random forests and boosting trees
]


#slide(title: "Ensemble models")[
  = Bagging: Bootstrap AGGregatING

  Bootstrap resampling (random sampling with replacement) proposed by @breiman1996bagging

  Built upon Bootstrap, introduced by @efron1992bootstrap to estimate the variance of an estimator.

  Bagging is used in machine learning to reduce the variance of a model prone to overfitting

  Can be used with any model
]

#slide(title: "Random forests: Bagging with classification trees
")[
  #set align(top)
  #side-by-side(
    [
      #only((1, 2))[
        #figure(image("img/3-flexible_models/bagging0.svg", width: 80%))
      ]
      #only(3)[
        #figure(image("img/3-flexible_models/bagging0_cross.svg", width: 80%))
      ]
    ],
    [
      #only(1)[
        #figure(image("img/3-flexible_models/bagging.svg", width: 90%))
      ]
      #only(2)[
        #figure(image("img/3-flexible_models/bagging_line.svg", width: 90%))
        #figure(image("img/3-flexible_models/bagging_trees.svg", width: 90%))
      ]
      #only(3)[
        #figure(image("img/3-flexible_models/bagging_cross.svg", width: 90%))
        #figure(image("img/3-flexible_models/bagging_trees_predict.svg", width: 90%))
        #figure(image("img/3-flexible_models/bagging_vote.svg", width: 90%))
      ]
    ],
  )
]
#slide(title: "Random forests: Bagging with regression trees")[
  #only(1)[
    #figure(image("img/3-flexible_models/bagging_reg_data.svg", width: 50%))
  ]
]
#slide(title: "Random forests: Bagging with regression trees")[
  #set align(top)
  #only(1)[
    #figure(image("img/3-flexible_models/bagging_reg_grey.svg", width: 100%))
  ]
  #only((2, 3))[
    #figure(image("img/3-flexible_models/bagging_reg_grey_fitted.svg", width: 100%))
  ]
  #h(1em)
  #side-by-side(
    [
      #uncover((1, 2, 3))[
        === - Select multiple subsets of the data
      ]
      #uncover((2, 3))[
        === - Fit one model on each
      ]
      #uncover(3)[
        === - Average the predictions
      ]
    ],
    [
      #only(3)[
        #figure(image("img/3-flexible_models/bagging_reg_blue.svg", width: 60%))
      ]
    ],
  )
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

  = Take away

  - Bagging and random forests fit trees independently
  - Each deep tree overfits individually
  - Averaging the tree predictions reduces overfitting
]

#slide(title: "Boosting: Adaptive boosting")[
  == Boosting use multiple iterative models

  - Use of simple underfitting models: eg. shallow trees

  - Each model corrects the errors of the previous one

  == Two examples of boosting

  - Adaptive boosting (AdaBoost): reweight mispredicted samples at each step @friedman2000additive

  - Gradient boosting: predict the negative errors of previous models at each step @friedman2001greedy
]

#slide(title: "Boosting: Adaptive boosting")[
  #side-by-side(
    [
      #only(1)[
        #figure(image("img/3-flexible_models/boosting0.svg"))
      ]
      #only(2)[
        == First prediction:
        #figure(image("img/3-flexible_models/boosting1.svg"))
      ]
      #only(3)[
        #figure(image("img/3-flexible_models/boosting2.svg"))
      ]
      #only(4)[
        #figure(image("img/3-flexible_models/boosting3.svg"))
      ]
    ],
    [
      #only(2)[
        #figure(image("img/3-flexible_models/boosting_trees1.svg"))
      ]
      #only(3)[
        #figure(image("img/3-flexible_models/boosting_trees2.svg"))
      ]
      #only(4)[
        #figure(image("img/3-flexible_models/boosting_trees3.svg"))
        == At each step, AdaBoost weights mispredicted samples
      ]
    ],
  )
]

#slide(title: "Gradient boosting")[

]

#slide(title: "Gradient boosting: how are the iterative learners chosen?")[
  == Boosting formulation

  $F_m(x) = F_(m-1)(x)+h_m(x)$ with $F_(m-1)$ the previous estimator, $h_m$, new week learner.

  == Minimization problem

  $h_m = argmin(h) (L_m)=argmin(h) sum_(i=1)^n l(y_i, F_(m-1)(x_i)+h(x_i))$

  #uncover(3)[
    - $l(y_i, F_(m-1)(x_i) + h(x_i)) = l(y_i, F_(m-1)(x_i)) + h_m(x_i) [(diff l (y_i, F(x_i))) / (diff F(x_i))]_(F=F_(m-1))$
  ]
  #only((2, 3))[
    #def_box(title: "💡Taylor expansion")[
      For $l(dot)$ differentiable: $l(y+h) approx l(y) + h (diff l) / (diff y) (y)$
    ]
  ]
]
]

#slide(title: "Faster gradient boosting with binned features")[
  == 😭 Gradient boosting is slow when N>10,000

  == 🚀 HistGradientBoosting

  - Discretize numerical features into 256 bins: less costly for tree splitting
  - Multi core implementation
  - Much much faster
]


#slide(title: "Take away for ensemble models")[


  #table(
    columns: 2,
    [
      [strong("Bagging"), strong("Boosting")],
      ["fit trees independently", "fit trees sequentially"],
      ["each deep tree overfits", "each shallow tree underfits"],
      ["averaging the tree predictions reduces overfitting", "sequentially adding trees reduces underfitting"],
    ],
  )

]


#new-section-slide("A word on other families of models")


#slide(title: "Other well known families of models")[

  = Generalized linear models

  = Kernel methods: Support vector machines, Gaussian processes

  = Deep neural networks
]


#slide(title: "Why not use deep learning everywhere?")[

  - Success of deep learning (aka deep neural networks) in image, speech recognition and text

  - 🤔 Why not so used in econometrics?

  #uncover(2)[
    == Deep learning needs a lot of data (typically $N approx 1$ million)

    == Do we have this much data in econometrics?
  ]
]


#slide(title: "Answer 1: Limited data settings")[
  - Typically #only(1)[in economics] (but also everywhere), we have a limited number of observations

  #figure(
    image("img/ML_1/2020_kdd_dataset_sizes.png", width: 65%),
    caption: [Typical dataset are mid-sized. This does not change with time. #footnote("https://www.kdnuggets.com/2020/07/poll-largest-dataset-analyzed-results.html")],
  )

]

#slide(title: "Answer 2: Deep learning underperforms on data tables")[

  === Tree-based methods outperform tailored deep learning architectures @grinsztajn2022tree

  #figure(image("img/ML_1/tree_outperforms_dl.png", width: 83%))
]

#slide(title: "Nuance: recent work on LLM and pre-trained techniques for tabular learning")[

  == Some references:

  - #link("https://skrub-data.org/stable/", "Skrub python library"): data-wrangling and encoding (same people than sklearn)
  - @kim2024carte: CARTE: pretraining and transfer for tabular learning
  - @grinsztajn2023vectorizing : Vectorizing string entries for data processing on tables: when are larger language models better?
]



#let bibliography = bibliography("biblio.bib", style: "apa")

#set align(left)
#set text(size: 20pt, style: "italic")
#slide[
  #bibliography
]
