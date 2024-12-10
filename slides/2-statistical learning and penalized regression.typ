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

//colors for the bias-variance trade-off
#let c_bayes = red.opacify(-50%)
#let c_bias = purple.opacify(-50%)
#let c_variance = green.opacify(-50%)

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
  extra: [A lot of today's content is taken from the excellent #link("https://inria.github.io/scikit-learn-mooc/toc.html", "sklearn mooc") @loic_esteve_2022_7220307],
  date: "January 10, 2025",
)


#slide(title: "Today's program")[

- Last session: importance of causal variable status
- Today: #alert[predictive inference] in high dimensions
  - Statistical learning basics
  - Penalized linear regression for predictive inference
  - Hands-on with scikit-learn

#uncover(2)[
- Next session:
  - Flexible models: Trees, Random Forests, Gradient Boosting
  - Practical scikit-learn
]
]


#slide(title: "Table of contents")[
  #metropolis-outline
]
#new-section-slide("Statistical learning framework")

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

#slide(title: "Statistical learning, two classes of problems")[
  

  #side-by-side[
    #figure(
      image("img/ML_1/linear_regression.png", width: 80%),
    )
  ][
    === Regression

   - The outcome is continuous: eg. wage prediction

   - The error is often measured by the mean squared error (MSE):
   #eq[$text("MSE") = EE[(Y - hat(f)(X))^2]$]
  ]
]
  

#slide(title: "Statistical learning, two classes of problems")[
  #side-by-side[
    #figure(
      image("img/ML_1/linear_classification.png", width: 80%),
    )
  ][
  === Classification

    - Outcome is categorical: eg. diagnosis, loan default, ...

    - Error is often measured with accuracy: 
    #eq[$text("Misclassification rate") = EE[bb(1)(Y != hat(f)(X))]$] with $hat(f) in {0, 1}$ for binary classification

]
]

#new-section-slide("Motivation: why prediction?")

#slide(title: "Why do we need prediction for ?")[
 
 == Statistical inference

   - Goal: infer some intervention effect with a causal interpretation
   - Require to regress "away" the relationship between the treatment or the outcome and the confounders #alert[-> more on this in sessions on Double machine learning.]
 
  == Predictive inference
  
   - Some problems in economics requires accurate prediction without a causal interpretation @kleinberg2015prediction 
   - Eg. Stratisfying on a risk score (loan, preventive care, ...)
]

#slide(title: "Do we need more than linear models?")[
  
  Let: 
    - $p$ is the number of features
    - $n$ is the number of observations

  == Maybe no

  - Low-dimensional data: $n>>p$

  - No non-linearities, no or few interactions between features

  == Maybe yes

  - High-dimensional data: ie. $p >> n$

  - Non-linearities, many interactions between features
]

#slide(title: "Do we need more than linear models?")[
  
  == When do we have "high-dimension"?

  - Is $p >> n$ a common setting in economics?
  - Consider the wage dataset:
    - $n = 5150$ individuals
    - $d=18$ variables
    
    #uncover(2)[
    - But, categorical variables, non-linearities and interactions increase the real number of features: 
      - non-linearities: add polynomials of degree 2: $p=2 times 18=36$
      - interactions: 
        - Of degree 2: $binom(d, 2)=binom(18, 2)=153$
        - All interactions: $2^(d)=2^(18)-18-1=262 125$
    ]
]

#slide(title: "Is this common?")[
  == Yes

  - Categorical or text data are increasingly common

  - Image data is high-dimensional by nature

  - Automation of data collection and storage leads to more collections of variables
  
  == Some examples:

  - The #link("https://www.census.gov/data/datasets/time-series/demo/cps/cps-basic.html", "Current Population Survey (CPS)") dataset has hundreds of variables, many of which are categorical
  
  - The #link("https://health-data-hub.shinyapps.io/dico-snds/", "Système National des Données de Santé (SNDS)") in France collects all reimbursments : many hundreds of variables, many of which are categorical

  - The #link("https://ssphub.netlify.app/post/parquetrp/", "population referencement dataset") from INSEE lists 800 pairs of (variables, categories).
]


#new-section-slide("Statistical learning theory and intutions")

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

#slide(title: "Train vs test error: simple models")[
  #side-by-side[
    === Measure the errors on the training data = fitting
    #figure(
      image("img/pyfigures/ols_simple_w_r2.svg", width: 80%),
    )
  ][
     === Measure the performances on test data = generalization
    #figure(
      image("img/pyfigures/ols_test_w_r2.svg", width: 80%),
    )
  ]
  #emoji.party Here, no problem of overfitting: train vs test error are similar.
]


#slide(title: "Train vs test error: flexible models")[
  #side-by-side[
    === Measure the errors on the training data = fitting
    #figure(
      image("img/pyfigures/splines_cubic_w_r2.svg", width: 80%),
    )
  ][
     === Measure the performances on test data = generalization
    #figure(
      image("img/pyfigures/splines_test_w_r2.svg", width: 80%),
    )
  ]
  \u{1F62B} Overfitting: the model is too complex and captures noise.
]
  



#slide(title: "How to choose the complexity of the model?")[
   #grid(
    columns: (auto, auto),
    gutter: 3pt,
    image("img/ML_1/ols_simple_test.svg", width: 70%),
    image("img/ML_1/splines_cubic_test.svg", width: 70%),
  )

  #only(2)[
    This trade-off is is called #alert[Bias variance trade-off]. 

    - Let's recover it in the context of statistical learning theory.
]
]

#slide(title: "Empirical Risk Minimization")[

  - Define a loss function $ell$ that defines proximity between the predicted value $hat(y) = f(x)$ and the true value $y$: $ell(f(x), y)$
  
  - Usually, for continuous outcomes, the squared loss is used: $ell(f(x), y) = (f(x) - y)^2$
  
  - We choose among a (finite) family of functions $f in cal(F)$, the best possible function $f^star$ minimizes the #alert[risk or expected loss] $cal(E)(f) = EE[(f(x) - y)^2]$:
  
  #eq[$f^star = text("argmin")_(f in cal(F)) EE[(f(x)- y)^2]$] 
]

#slide(title:"Empirical risk minimization: estimation error")[
   - In finite sample regimes, the expectation is not accessible since we only have access to a finite number of data pairs
  
  - In practice, we minimize the #alert[empirical risk] or average loss $R_(text("emp"))= sum_(i=1)^n (f(x_i) - y_i)^2$:
  
  #eq[$hat(f) = text("argmin")_(f in cal(F)) sum_(i=1)^n (f(x_i) - y_i)^2$]

  - This creates the #highlight(fill:c_variance)[estimation error], related to sampling noise: 
  #eq[$cal(E)(hat(f)) - cal(E)(f^(star)) = EE[(hat(f)(x) - y)^2] - EE[(f^(star)(x) - y)^2] >= 0$]
]

#slide(title:"Empirical risk minimization: estimation error illustration")[
  = High #highlight(fill: c_variance)[estimation error] means overfit
  
  #grid(
    columns: (auto, auto),
    gutter: 3pt,
    align: center,
   image("img/ML_1/polynomial_overfit_simple_legend.svg", width: 90%),
    [
      #set align(left)
     == Model is too complex
    - The model is able to recover the true generative process
    - But its flexibility captures noise
    == Too much noise
    == Not enough data
    ],
  )
  
]

#slide(title: "Bayes error rate: Randomness of the problem")[

- Interesting problems exhibit randomness #linebreak()
$y=g(x)+ e$ with $E(e|x)=0$ and $text("Var")(e|x)=sigma^2$

- The best possible estimator is $g(dot)$, yielding the #highlight(fill:c_bayes)[Bayes error], the unavoidable error:
  #eq[$cal(E)(g) = EE[(g(x) + e - g(x))^2] = EE[e^2]$]

]

#slide(title:"Empirical risk minimization: approximation error")[ 

  - In practice you don't know the class of function in which the true function lies : $y approx g(x)$ : Every model is wrong ! 
  
  - You are choosing the best possible function in the class of functions you have access to: $f^star in cal(F)$ eg. linear models, polynomials, trees, ...
  
  - This creates the #highlight(fill:c_bias)[approximation error]: 
  #eq[$cal(E)(f(star)) - cal(E)(g) = EE[(f^(star)(x) - y)^2] - EE[(g(x) - y)^2] >= 0$] 
]


#slide(title:"Empirical risk minimization: approximation error illustration")[
  = High #highlight(fill: c_bias)[approximation error] means underfit
  #h(1em)
  #grid(
    columns: (auto, auto),
    gutter: 3pt,
    align: center,
   image("img/ML_1/polynomial_underfit_simple.svg", width: 90%),
    [
      #set align(left)
     == Model is too simple for the data
    - its best fit does not approximate the true generative process
    - Yet it captures little noise
    == Low noise
    == Rapidly enough data to fit the model
    ],
  )
  
]


#slide(title: "Bias variance trade-off: Putting the pieces together")[
  == Decomposition of the empirical risk of a fitted model $hat(f)$
  
  #h(1em)
  #eq[
    $cal(E)(hat(f)) = 
    #rect(fill:c_bayes,inset:15pt)[$underbrace(cal(E)(g), "Bayes error")$] + 
    #rect(fill:c_bias,inset:15pt)[$underbrace(cal(E)(f^(star)) - cal(E)(g), "approximation error")$] + 
    #rect(fill:c_variance, inset:15pt)[$underbrace(cal(E)(hat(f)) - cal(E)(f^(star)), "estimation error")$]$
  ]

#only(2)[
  == Controls on this trade-off

  - Increase/decrease the size of the hypothesis family : $cal(F)$ ie. more or less complex models.

  - Increase your sample size: $n$ ie. more observations.
]
]

#slide(title: "Train vs test error: increasing complexity")[

#set align(center)

#only(1)[
#side-by-side(
  image("img/ML_1/polynomial_overfit_test_1.svg", width: 80%),
  image("img/ML_1/polynomial_validation_curve_1.svg", width: 80%),
)
]
#only(2)[
#side-by-side(
  image("img/ML_1/polynomial_overfit_test_2.svg", width: 80%),
  image("img/ML_1/polynomial_validation_curve_2.svg", width: 80%),
)
]
#only(3)[
#side-by-side(
  image("img/ML_1/polynomial_overfit_test_5.svg", width: 80%),
  image("img/ML_1/polynomial_validation_curve_5.svg", width: 80%),
)
]
#only(4)[
#side-by-side(
  image("img/ML_1/polynomial_overfit_test_9.svg", width: 80%),
  image("img/ML_1/polynomial_validation_curve_15.svg", width: 80%),
)
]

#only(5)[
  #figure(
    image("img/ML_1/polynomial_validation_curve_15.svg", width: 50%)
  )
  #set text(size: 1.3em) 
  #align(left)[
  #h(6em)
  #highlight(fill: rgb("#440154").opacify(-50%))[*Underfit*] 
  #h(0.2em) #highlight(fill: rgb("#7ad151").opacify(-50%))[*Sweet Spot*] 
  #h(0.2em)  #highlight(fill: rgb("#440154").opacify(-50%))[*Overfit*]
  ]
]
]

#slide(title: "Varying sample size")[

#set align(center)
#set text(size: 2em) 

#only(1)[
#side-by-side(
  image("img/ML_1/polynomial_overfit_ntrain_42.svg", width: 80%),
  image("img/ML_1/polynomial_learning_curve_42.svg", width: 80%),
)
  #align(left)[
  #h(3em)
  #highlight(fill: rgb("#440154").opacify(-50%))[*Overfit*] 
  ]
]


#only(2)[
#side-by-side(
  image("img/ML_1/polynomial_overfit_ntrain_145.svg", width: 80%),
  image("img/ML_1/polynomial_learning_curve_145.svg", width: 80%),
)
]

#only(3)[
#side-by-side(
  image("img/ML_1/polynomial_overfit_ntrain_1179.svg", width: 80%),
  image("img/ML_1/polynomial_learning_curve_1179.svg", width: 80%),
)
  #align(left)[
  #h(3em)
  #highlight(fill: rgb("#7ad151").opacify(-50%))[*Sweet spot?*] 
  ]
]

#only(4)[
#side-by-side(
  image("img/ML_1/polynomial_overfit_ntrain_6766.svg", width: 80%),
  image("img/ML_1/polynomial_learning_curve_6766.svg", width: 80%),
)
  #align(left)[
  #h(3em)
  #highlight(fill: rgb("#440154").opacify(-50%))[*Diminishing returns?*] 
  ]
]
#set text(size: 22pt) 
#set align(left)

#only(5)[
#side-by-side(
  image("img/ML_1/polynomial_overfit_ntrain_6766.svg", width: 80%),
  [
    The error of the best model trained on unlimited data. 

    Here, the data is generated by a polynomial of degree 9.

    We cannot do better.

    Prediction is limited by noise.
  ]
)
]
]



#slide(title: "Remaining of this session (and the next)")[

  = Explore common families of models suited to tables data

  == Today

  - Penalized linear regression: Lasso and Ridge

  - Hands-on with scikit-learn

  == Next session

  - Flexible models: Trees, Random Forests, Gradient Boosting

  - Practical model selection: Cross-validation

  - Practical scikit-learn
]

#new-section-slide("Lasso for predictive inference")

#slide(title:"Linear model reminder: Linear regression")[

  #only(1)[

  $y$ is a linear combination of the features $x in RR^p$
  
  ]
  #eq[$Y_i = X_i^T beta_0 + epsilon_i$]

  #only(1)[ 
  - $epsilon$ the random variable of the error term.

  - $beta_0 in RR^(p times 1)$ the _true_ coefficients.

  Usually, we assume that the errors are normally distributed and independent of X_i: 
  
  $epsilon_i tilde cal(N)(0, sigma^2)$ and $epsilon_i tack.t.double X_i$
  ]
  #only(2)[
    #figure(
      image("img/ML_1/linear_fit_red.svg", width: 50%),
    )
  ]
  Model are typically fitted by linear algebra methods @hastie2009elements.
]

#slide(title:"Linear model reminder: Linear regression")[
  TODO: Metrics
]

#slide(title:"Linear regression: Two dimension illustration")[
  #figure(
    image("img/ML_1/lin_reg_3D.svg", width: 50%),
  )
]


#slide(title:"Linear model reminder: classification, logistic regression")[
  #only(1)[ 

  The logit of the probability of the outcome is a linear combination of the features $X_i in RR^p$:

  #eq[$ln(p(Y_i=1|X_i)/p(Y_i=0|X_i)) = X_i^T beta_0$]

  which is equivalent to:
  ]
  #eq[$p(Y_i=1|X_i, beta_0) \u{225D} p(X_i, beta_0) = 1/(1+exp(-X_i^T beta_0))$]
  #only(1)[ 

  The statistical model is a bernoulli: $B(p(x, beta_0))$

  Models are fitted by maximizing the likelihood by iterative optimization @hastie2009elements #footnote[eg. coordinate descents (liblinear), second order descent (Newton's method), gradient descent (SAG)...].
  ]
   #only(2)[
    #figure(
      image("img/ML_1/logistic_color.svg", width: 50%),
    )
  ]
]

#slide(title:"Linear model reminder: classification, logistic regression")[
  TODO: Metrics
]

#slide(title:"Logistic regression: Two dimension illustration")[
 #figure(
    image("img/ML_1/logistic_2D.svg", width: 50%),
  )
]

#slide(title: "Linear models are not suited to all data")[
  #set align(center)

  #side-by-side(
    [
      #image("img/ML_1/lin_separable.svg", width: 80%)
      Almost linearly separable data.
    ],

    [
      #image("img/ML_1/lin_not_separable.svg", width: 80%)
      Data not linearly separable.
      ],
  )
]

#slide(title: "Linear model pros and cons")[

#only(1)[
  == Pros

  - Converge quickly

  - Hard to beat when n_features is large but we still have n_samples >> n_features

  - Linear models work well if
  
    - the classes are (almost) linearly separable
    - or the outcome is (almost) linearly related to the features.
] 
#only((2,3))[
  == Cons

  Sometimes

    - the best decision boundary to separate classes is not well approximated by a straight line.

    - there are important non-linear relationships between the features and the outcome.
]

#only(3)[
  #emoji.lightbulb Either use non-linear models, or perform transformations on the data, to engineer new features.
]
]

#slide(title:"Transformation of the features: Example")[

#only(1)[
#side-by-side(
  [
    #figure(
      image("img/pyfigures/2_linear_regression_non_linear_link.svg", width: 80%),
  )
  ],
  [
    Non-linear relationship between the features and the outcome:

    $Y = X^3 - 0.5 times X^2 + epsilon$
  ]
)
]


#only(2)[
#side-by-side(
  [
    #figure(
      image("img/pyfigures/2_linear_regression_non_linear_link_linear.svg", width: 100%),
  )
  ],
  [
  
    Vanilla Linear regression fails to capture the relationship.

  ]
)
]

#only(3)[
#side-by-side(
  [
    #figure(
      image("img/pyfigures/2_linear_regression_non_linear_link_polynomial.svg", width: 100%),
  )
  ],
  [
  
  Solution: 
  
   - Expand the feature space with polynoms of the features:
    
    $X = [X, X^2, X^3]$
  
  - Run a linear regression on the new feature space.

  $Y = [X, X^2, X^3]^T hat(beta)$
  ]
)
]
]


#slide(title:"Sparsity")[

]

#slide(title:"Lasso: intuition")[

]

#slide(title:"Post-Lasso: intuition")[

]

#slide(title:"Pitfalls on using Lasso for variable selections")[

]


#slide(title:"Ridge regression")[

]

#slide(title:"Elastic net")[

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


#slide(title: "Take home messages: Bias-variance trade-off")[
 == High bias == underfitting

  - systematic prediction errors
  - the model prefers to ignore some aspects of the data
  - mispecified models

== High variance == overfitting:

 - prediction errors without obvious structure
 - small change in the training set, large change in model
 - unstable models
]

#slide(title: "Take home messages: Lasso and Ridge")[

]

#new-section-slide("Practical session")


#let bibliography = bibliography("biblio.bib", style: "apa")

#set align(left)
#set text(size: 20pt, style: "italic")
#slide[
  #bibliography
]