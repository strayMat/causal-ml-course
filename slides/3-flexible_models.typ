// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.3.1": *
#import "@preview/embiggen:0.0.1": * // LaTeX-like delimiter sizing for Typst
#import "@preview/codly:1.1.1": * // Code highlighting for Typst
#show: codly-init

#import "@preview/codly-languages:0.1.3": *
#codly(languages: codly-languages)

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
    - Proof that it selects the best model (averaging on the folds): @lecue2012oracle
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
  - Either refit on full data the model with the best hyper-parameters on the full data
  - Or use the aggregation of outputs from the cross-validation of the best model:
  $hat(y) = 1/K sum_(k=1)^K hat(y)_k$ where $hat(y)_k$ is the prediction of the model trained on the $k$-th fold
]

#slide(title: "Naive cross-validation to select AND estimate the best performances")[

]

#slide(title: "Nested cross-validation to select the best model")[

]

#new-section-slide("Flexible models: Tree, random forests and boosting")


#slide(title: "Tree for predictive inference")[

]


#slide(title: "Random Forests for predictive inference")[

]

#slide(title: "Boosting")[

]

#slide(title: "Ensemble models")[]

#new-section-slide("A word on other families of models")

#slide(title: "Why not use deep learning everywhere?")[

  - Success of deep learning (aka deep neural networks) in image, speech recognition and text

  - Why not so used in econometrics?

    == Deep learning needs a lot of data (typically $N approx 1$ million)

    - Do we have this much data in econometrics?
]


#slide(title: "Limited data settings")[
  - Typically #only(1)[in economics] (but also everywhere), we have a limited number of observations

  #figure(
    image("img/ML_1/2020_kdd_dataset_sizes.png", width: 65%),
    caption: [Typical dataset are mid-sized. This does not change with time. #footnote("https://www.kdnuggets.com/2020/07/poll-largest-dataset-analyzed-results.html")],
  )

]

#slide(title: "Deep learning underperforms on data tables")[

  == Tree-based methods outperform tailored deep learning architectures @grinsztajn2022tree

  #figure(
    image("img/ML_1/tree_outperforms_dl.png", width: 83%),
    caption: "DAG for a RCT: the treatment is independent of the confounders",
  )
]


#slide(title: "Other well known families of models")[

  = Generalized linear models

  = Support vector machines

  = Gaussian processes

]



#let bibliography = bibliography("biblio.bib", style: "apa")

#set align(left)
#set text(size: 20pt, style: "italic")
#slide[
  #bibliography
]
