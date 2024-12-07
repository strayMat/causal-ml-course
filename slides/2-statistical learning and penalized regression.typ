// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.3.1": *
#import "@preview/embiggen:0.0.1": *
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
  subtitle: "Statistical learning and penalized regression",
  date: "January 10, 2025",
)


#slide(title: "Today's program")[

- Last session: importance of causal variable status
- Today: #alert[predictive inference] in high dimensions
  - Statistical learning basics
  - Penalized regression for predictive inference
#uncover(2)[
- Next session:
  - Flexible models: Trees, Random Forests, Gradient Boosting
  - Practical scikit-learn
]
]


#slide(title: "Table of contents")[
  #metropolis-outline
]
#new-section-slide("Settings: statistical learning")

#slide(title: "Statistical learning, ie. predictive inference")[
  == Goal

  - Predict the value of an outcome based on one or more input variables.

  == Setting

  - Data: n pairs of (features, outcome), $(x_i, y_i) in cal(X) times cal(Y)$ identically and independently distributed (i.i.d.) from an unknown distribution $P$.
  - Goal: find a function $hat(f): cal(X) -> cal(Y)$ that approximates the true value of $y$ ie. for a new pair $(x, y)$, we should have:  
  
  #eq[$hat(y)=hat(f)(x) approx y$]

  == Vocabulary

  Finding the appropriate model $hat(f)$ is called learning, training or fitting the model.
]

#slide(title: "Statistical learning, two types of problems")[
  
  === Regression

   - The outcome is continuous: eg. wage prediction

   - The error is often measured by the mean squared error (MSE):
   #eq[$text("MSE") = EE[(Y - hat(f)(X))^2]$]

  === Classification

    - The outcome is categorical: eg. diagnosis, loan default, ...

    - The error is often measured by the accuracy: 
    #eq[$text("Misclassification rate") = EE[bb(1)(Y != hat(f)(X))]$]

]

#new-section-slide("Motivation: Why prediction?")

#slide(title: "Why do we need prediction for ?")[
 
 == Statistical inference

   - Goal: infer some intervention effect with a causal interpretation
   - Require to regress "away" the relationship between the treatment or the outcome and the confounders #alert[-> more on this in sessions on Double machine learning.]
 
  == Predictive inference
  
   - Some problems in economics requires accurate prediction @kleinberg2015prediction without a causal interpretation
   - Eg. Stratisfying on a risk score (loan, preventive care, ...)
]

#slide(title: "Do we need more than linear models?")[
  
  Let: 
    - $p$ is the number of features
    - $n$ is the number of observations

  == Maybe no

  - Low-dimensional data: $n>>p$

  - High predictive performances 

  == Maybe yes

  - High-dimensional data: ie. $p >> n$
  - Poor predictive performances
]

#slide(title: "Do we need more than linear models?")[
  
  == When do we have "high-dimension"?


  - $p >> n$ is a common setting in economics
  
]




#new-section-slide("Statistical learning theory")

#slide(title: "Under vs. overfitting")[
   
  = Which data fit do you prefer?

   #grid(
    columns: (auto, auto),
    gutter: 3pt,
    image("img/ML_1/linear_ols.svg", width: 70%),
    image("img/ML_1/linear_splines.svg", width: 70%),
  )
]

#slide(title: "Under vs. overfitting")[

  = Which data fit do you prefer? (new data incoming)

   #grid(
    columns: (auto, auto),
    gutter: 3pt,
    image("img/ML_1/linear_ols_test.svg", width: 70%),
    image("img/ML_1/linear_splines_test.svg", width: 70%),
  )

  - Answering this question might be hard. 
  - Goal: create models that generalize.
  - The good way of framing the question is: #alert[how will the model perform on new data?]
]

#slide(title: "Under vs. overfitting")[

  = Which data fit do you prefer? New example! 

   #grid(
    columns: (auto, auto),
    gutter: 3pt,
    image("img/ML_1/ols_simple_test.svg", width: 70%),
    image("img/ML_1/splines_cubic_test.svg", width: 70%),
  )

  #only(2)[
    This trade-off is is called #alert[Bias variance trade-off]. 

    - Let's recover this trade-off in the context of statistical learning theory.
]
]

#slide(title: "Empirical Risk Minimization")[

  - Define a loss function $ell$ that defines proximity between the predicted value $hat(y) = f(x)$ and the true value $y$: $ell(f(x), y)$
  
  - Usually, for continuous outcomes, the squared loss is used: $ell(f(x), y) = (f(x) - y)^2$
  
  - We choose among a (finite) family of functions $f in cal(F)$, the best possible function $f^star$ minimizes the #alert[risk or expected loss] $R = EE(ell)$:
  
  #eq[$f^star = text("argmin")_(f in cal(F)) EE[(y - y)^2]$] 
]

#slide(title:"")[
   - In finite sample regimes, the expectation is not accessible since we only have access to a finite number of data pairs
  
  - In practice, we minimize the #alert[empirical risk] or average loss $R_(text("emp"))= sum_(i=1)^n (f(x_i) - y_i)^2$:
  
  #eq[$hat(f) = text("argmin")_(f in cal(F)) sum_(i=1)^n (f(x_i) - y_i)^2$]
]

#slide(title: "Bayes error rate: Randomness of the problem")[
In most interesting problems, there is some randomness: ie. $y=g(x)+ e$ with $E(e|x)=0$ and $text("Var")(e|x)=sigma^2$
]

#slide(title: "Bias variance trade-off")[
In most interesting problems, there is some random
]


#new-section-slide("Lasso for predictive inference")

#slide(title: "Bias-variance trade-off, take home messages")[
 == High bias == underfitting

  - systematic prediction errors
  - the model prefers to ignore some aspects of the data
  - mispecified models

== High variance == overfitting:

 - prediction errors without obvious structure
 - small change in the training set, large change in model
 - unstable models
]

#slide(title: "")[

]

#slide(title: "")[

]


#slide(title: "")[

]

#slide(title: "")[

]

#new-section-slide("A word on deep learning")

#slide(title: "Why not use deep learning everywhere?")[

- Success of deep learning in image, speech recognition and text

- Why not so used in economics?
]


#slide(title: "Limited data settings")[
  - Typically #only(1)[in economics] #only(1)[everywhere], we have a limited number of observations

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

#slide[

  = Resources

  - https://web.stanford.edu/~swager/stats361.pdf
  - https://www.mixtapesessions.io/
  - https://alejandroschuler.github.io/mci/
  - https://theeffectbook.net/index.html
]

#let bibliography = bibliography("biblio.bib", style: "apa")

#set align(left)
#set text(size: 20pt, style: "italic")
#slide[
  #bibliography
]