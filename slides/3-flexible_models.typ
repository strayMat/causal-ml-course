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
    show: align.with(top)
    show: pad.with(left: 1em, top: 0em, right: 1em, bottom: 0em) // super important to have a proper padding (not 1/3 of the slide blank...)
    set text(fill: m-dark-teal)
    body
  }

  logic.polylux-slide(content)
}

// Use #polylux-slide to create a slide and style it using your favourite Typst functions
// #polylux-slide[
//   #align(horizon + center)[
//     = Very minimalist slides

//     A lazy author

//     July 23, 2023
//   ]
// ]

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

#uncover(2)[- 🤔 But... How to select the best model? the best hyperparameters?]
]


#slide(title: "Table of contents")[
  #metropolis-outline
]

#new-section-slide("Model evaluation and selection with cross-validation")

#slide(title: "A closer look at model evaluation: Wage example")[
  == Example with the Wage dataset

  - Raw dataset: (N=534, p=11)

  #only(1)[
      #figure(image(
      "img/3-flexible_models/wage_head.png"
    ))
    ]

  #uncover((2,3))[
    - Transformation: encoding of categorical data and scaling of numerical data
  ]

  #only(2)[
    #figure(image(
      "img/3-flexible_models/wage_transformed_head.png"
    ))
  ]

  #uncover(3)[
    - Regressor: Lasso with regularization parameter ($alpha=10$), the final pipeline is:
  ]
  #only(3)[
    #figure(image(
      "img/3-flexible_models/wage_pipeline.png", width: 50%
    ))
  ]
]

#slide(title: "Repeated train/test splits")[
  #only(1)[
    == Splitting once: In red, the training set, in blue, the test set
  
    #figure(
      image(
        "img/pyfigures/train_test_split_visualization_seed_0.png", width: 65%
      ),
    )
  ]
  #only(2)[
    == But we could have chosen another split ! Yielding a different MAE
    
    #figure(
      image(
        "img/pyfigures/train_test_split_visualization_seed_1.png", width: 65%
      ),
    )
  ]

  #only(3)[
    == And another split...
    
    #figure(
      image(
        "img/pyfigures/train_test_split_visualization_seed_3.png", width: 65%
      ),
    )
  ]

  #only(4)[
    == Splitting ten times
    
    #figure(
      image(
        "img/pyfigures/train_test_split_visualization_seed_9.png", width: 65%
      ),
    )
  ]
]

#slide(title: "Repeated train/test splits = Cross-validation")[

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
    - It is a more robust way to evaluate the model's performance:
    - We get a more robust estimate by taking the mean over the repetitions
    - We get a better idea of the variability of the model's performance: similar to bootstrapping (but different)
  ]
]

#slide(title: "How to select a model?")[

]
#new-section-slide("Tree, random forests and boosting")

#slide(title:"Random Forests for predictive inference")[

]

#slide(title:"Boosting")[

]

#slide(title:"Ensemble models")[]

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

  - Generalized linear models: 

  - Support vector machines:

  - Gaussian processes: 
  
]



#let bibliography = bibliography("biblio.bib", style: "apa")

#set align(left)
#set text(size: 20pt, style: "italic")
#slide[
  #bibliography
]