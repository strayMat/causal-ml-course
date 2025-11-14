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
