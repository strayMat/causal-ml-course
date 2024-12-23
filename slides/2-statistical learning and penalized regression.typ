// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.3.1": *
#import "@preview/embiggen:0.0.1": *
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

#let c_treated(body) = {
  set text(fill: blue)
  body
}

#let eq(body, size: 1.4em) = {
  set align(center)
  set text(size: size)
  body
}

// assumption box
//
//
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
    show: pad.with(
      left: 1em,
      top: 0em,
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
  subtitle: "Statistical learning and regularized linear models",
  extra: [A lot of today's content is taken from the excellent #link("https://inria.github.io/scikit-learn-mooc/toc.html", "sklearn mooc") @loic_esteve_2022_7220307],
  date: "January 10, 2025",
)


#slide(title: "Today's program")[

  - Last session: importance of causal variable status
  - Today: #alert[predictive inference] in high dimensions
    - Statistical learning basics
    - Regularized linear models for predictive inference
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
    #figure(image("img/ML_1/linear_regression.png", width: 80%))
  ][
    === Regression

    - The outcome is continuous: eg. wage prediction

    - The error is often measured by the mean squared error (MSE):
    #eq[$text("MSE") = EE[(Y - hat(f)(X))^2]$]
  ]
]


#slide(title: "Statistical learning, two classes of problems")[
  #side-by-side[
    #figure(image("img/ML_1/linear_classification.png", width: 80%))
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

#slide(title: "What high-dimension means: Is p >> n common in economics?")[

  == Characteristics of the dataset that can lead to high-dimensionality

  - Categorical variables with high cardinality, eg. job title, diagnoses...

  - Text data: eg. job description, medical reports...

  - Technical regressors to handle non-linearities, eg. polynomials, splines, log, ...
]

#slide(title: "What high-dimension means, concrete example")[

  - #link("https://ssphub.netlify.app/post/parquetrp/", "Population referencement dataset"), individual file (INSEE): n=19 735 576; p=88 #emoji.fingers

  - But many variables with cardinality: more than 555 pairs of (variable, category).

  - Adding interaction of degree 2: $binom(p, 2)=binom(555, 2) = (555 * 554) / 2 = 153 735$ features #emoji.face.sweat

  - Adding interactions of any degree: $2^(p) - p - 1 = 2^(555) - 554$ #emoji.face.explode
]


#slide(title: "Is this common? Yes")[

  - Categorical with high cardinality or text data are increasingly common.

  - Image data is high-dimensional by nature.

  - Automation of data collection and storage leads to more datasets and more variables.

]

#slide(title: "Some examples from area with high dimensional data")[

  == Some examples

  - The #link("https://www.census.gov/data/datasets/time-series/demo/cps/cps-basic.html", "Current Population Survey (CPS)") dataset has hundreds of variables, many of which are categorical

  - The #link("https://health-data-hub.shinyapps.io/dico-snds/", "Système National des Données de Santé (SNDS)") in France = healthcare claims : many hundreds of variables, many of which are categorical.

  #uncover(2)[
    == Other area

    - Country characteristics in cross-country wealth analysis,

    - Housing characteristics in house pricing/appraisal analysis,

    - Product characteristics at the point of purchase in demand analysis.
  ]
]

#new-section-slide("Statistical learning theory and intutions")

#slide(title: "Under vs. overfitting")[

  = Which data fit do you prefer?

  #grid(
    columns: (auto, auto),
    gutter: 3pt,
    image("img/ML_1/linear_ols.svg", width: 70%), image("img/ML_1/linear_splines.svg", width: 70%),
  )
]

#slide(title: "Under vs. overfitting")[

  = Which data fit do you prefer? (new data incoming)

  #grid(
    columns: (auto, auto),
    gutter: 3pt,
    image("img/ML_1/linear_ols_test.svg", width: 70%), image("img/ML_1/linear_splines_test.svg", width: 70%),
  )

  - Answering this question might be hard.
  - Goal: create models that generalize.
  - The good way of framing the question is: #alert[how will the model perform on new data?]
]

#slide(title: "Train vs test error: simple models")[
  #side-by-side[
    === Measure the errors on the training data = fitting
    #figure(image("img/pyfigures/ols_simple_w_r2.svg", width: 80%))
  ][
    === Measure the performances on test data = generalization
    #figure(image("img/pyfigures/ols_test_w_r2.svg", width: 80%))
  ]
  #emoji.party Here, no problem of overfitting: train vs test error are similar.
]


#slide(title: "Train vs test error: flexible models")[
  #side-by-side[
    === Measure the errors on the training data = fitting
    #figure(image("img/pyfigures/splines_cubic_w_r2.svg", width: 80%))
  ][
    === Measure the performances on test data = generalization
    #figure(image("img/pyfigures/splines_test_w_r2.svg", width: 80%))
  ]
  \u{1F62B} Overfitting: the model is too complex and captures noise.
]




#slide(title: "How to choose the complexity of the model?")[
  #grid(
    columns: (auto, auto),
    gutter: 3pt,
    image("img/ML_1/ols_simple_test.svg", width: 70%), image("img/ML_1/splines_cubic_test.svg", width: 70%),
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

#slide(title: "Empirical risk minimization: estimation error")[
  - In finite sample regimes, the expectation is not accessible since we only have access to a finite number of data pairs

  - In practice, we minimize the #alert[empirical risk] or average loss $R_(text("emp"))= sum_(i=1)^n (f(x_i) - y_i)^2$:

  #eq[$hat(f) = text("argmin")_(f in cal(F)) sum_(i=1)^n (f(x_i) - y_i)^2$]

  - This creates the #highlight(fill:c_variance)[estimation error], related to sampling noise:
  #eq[$cal(E)(hat(f)) - cal(E)(f^(star)) = EE[(hat(f)(x) - y)^2] - EE[(f^(star)(x) - y)^2] >= 0$]
]

#slide(title: "Empirical risk minimization: estimation error illustration")[
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

#slide(title: "Empirical risk minimization: approximation error")[

  - In practice you don't know the class of function in which the true function lies : $y approx g(x)$ : Every model is wrong !

  - You are choosing the best possible function in the class of functions you have access to: $f^star in cal(F)$ eg. linear models, polynomials, trees, ...

  - This creates the #highlight(fill:c_bias)[approximation error]:
  #eq[$cal(E)(f(star)) - cal(E)(g) = EE[(f^(star)(x) - y)^2] - EE[(g(x) - y)^2] >= 0$]
]


#slide(title: "Empirical risk minimization: approximation error illustration")[
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
    #figure(image("img/ML_1/polynomial_validation_curve_15.svg", width: 50%))
    #set text(size: 1.3em)
    #align(left)[
      #h(6em)
      #highlight(fill: rgb("#440154").opacify(-50%))[*Underfit*]
      #h(0.2em) #highlight(fill: rgb("#7ad151").opacify(-50%))[*Sweet Spot*]
      #h(0.2em) #highlight(fill: rgb("#440154").opacify(-50%))[*Overfit*]
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
      columns: (1fr, 1.7fr),
      image("img/ML_1/polynomial_overfit_ntrain_6766.svg", width: 100%),
      [
        The error of the best model trained on unlimited data.

        Here, the data is generated by a polynomial of degree 9.

        We cannot do better.

        Prediction is limited by noise: #highlight(fill:c_bayes)[Bayes error].
      ],
    )
  ]
]



#slide(title: "Remaining of this session (and the next)")[

  = Common model families suited to tabular data

  == Today

  - Regularized linear models: Lasso and Ridge

  - Hands-on with scikit-learn

  == Next session

  - Practical model selection: Cross-validation

  - Flexible models: Trees, Random Forests, Gradient Boosting

  - Practical scikit-learn
]

#new-section-slide("Regularized linear models for predictive inference")

#slide(title: "Reminder: Linear regression")[

  #only(1)[

    $y$ is a linear combination of the features $x in RR^p$

  ]
  #eq[$Y_i = X_i^T beta_0 + epsilon_i$]

  #only(1)[
    - $epsilon$ the random variable of the error term.

    - $beta_0 in RR^(p times 1)$ the _true_ coefficients.

    Usually, we assume that the errors are normally distributed and independent of $X_i$ :

    $epsilon_i tilde cal(N)(0, sigma^2)$ and $epsilon_i tack.t.double X_i$

    Model are typically fitted by linear algebra methods @hastie2009elements.
  ]
  #only(2)[
    #figure(image("img/ML_1/linear_fit_red.svg", width: 50%))
  ]

]

#slide(title: "Reminder: Linear regression")[
  == Common metrics

  - Mean Squared Error: $text("MSE") = 1/n sum_(i=1)^n (Y_i - hat(Y_i))^2$

  - R-squared, : $R^2 = 1 - (sum_(i=1)^(n)(Y_i - hat(Y_i))^2) / (sum_(i=1)^(n)(Y_i - dash(Y))^2)$ where $dash(Y) = 1/n sum_(i=1)^(n) Y_i$

  The proportion of variance explained by the model (perfect fit: $R^2=1$)

  - Mean absolute error: $text("MAE") = 1/n sum_(i=1)^n |Y_i - hat(Y_i)|$
]

#slide(title: "Linear regression: Illustration in two dimensions")[
  #figure(image("img/ML_1/lin_reg_3D.svg", width: 50%))
]


#slide(title: "Reminder: logistic regression for classification")[
  The logit of the probability of the outcome is a linear combination of the features $X_i in RR^p$:

  #eq[$ln(p(Y_i=1|X_i)/p(Y_i=0|X_i)) = X_i^T beta_0$]

  #uncover((2, 3))[
    Taking exponential of both sides, we get:
    #eq[$p(Y_i=1|X_i, beta_0) \u{225D} p(X_i, beta_0) = 1 / (1+exp(-X_i^T beta_0))$]

    The statistical model is a Bernoulli 🪙: $B(p(x, beta_0))$
  ]

  #uncover(3)[
    Model fitted by maximum likelihood with iterative optimization @hastie2009elements: eg. coordinate descents (liblinear), second order descent (Newton's method), gradient descent (SAG)...
  ]
]

#slide(title: "Reminder: classification, logistic regression")[
  #side-by-side(
    columns: (2fr, 1fr),
    [
      == Common metrics

      - $text("Accuracy") = 1 / n sum_(i=1)^n bb(1)(Y_i = hat(Y_i))$

      - Precision: $text("Precision") = text("TP") / (text("TP") + text("FP"))$

      - Recall: $text("Recall") = text("TP") / (text("TP") + text("FN"))$

      - Brier score loss: $text("BSL") = 1/n sum_(i=1)^n (Y_i - p_i)^2$
    ],
    [
      #image("img/ML_1/Precisionrecall.svg.png", width: 80%)
    ],
  )


]

#slide(title: "Logistic regression: Illustrations")[

  #side-by-side(
    [
      #image("img/ML_1/logistic_color.svg", width: 80%)
      Logistic regression in one dimension.
    ],
    [
      #image("img/ML_1/logistic_2D.svg", width: 80%)
      Logistic regression in two dimensions.
    ],
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
  #only((2, 3))[
    == Cons

    Sometimes

    - the best decision boundary to separate classes is not well approximated by a straight line.

    - there are important non-linear relationships between the features and the outcome.
  ]

  #only(3)[
    #emoji.lightbulb Either use non-linear models, or perform transformations on the data, to engineer new features.
  ]
]

#slide(title: "Transformation of the features: Example")[

  #only(1)[
    #side-by-side(
      [
        #figure(image("img/pyfigures/2_linear_regression_non_linear_link.svg", width: 80%))
      ],
      [
        Non-linear relationship between the features and the outcome:

        $Y = X^3 - 0.5 times X^2 + epsilon$
      ],
    )
  ]

  #only(2)[
    #side-by-side(
      [
        #figure(image("img/pyfigures/2_linear_regression_non_linear_link_linear.svg", width: 100%))
      ],
      [

        Vanilla Linear regression fails to capture the relationship.

      ],
    )
  ]

  #only(3)[
    #side-by-side(
      [
        #figure(image("img/pyfigures/2_linear_regression_non_linear_link_polynomial.svg", width: 100%))
      ],
      [

        Solution:

        - Expand the feature space with polynoms of the features:

          $X = [X, X^2, X^3]$

        - Run a linear regression on the new feature space.

        $Y = [X, X^2, X^3]^T hat(beta)$
      ],
    )
  ]
]


#slide(title: "Takeaway on feature expansion")[

  - Linear model can underfeat : when n_features small or the problem is not linearly separable.

  - Feature expansion is an easy way to capture non-linear relationships.

  #uncover(2)[
    But...
    - Linear models can also overfit !

    When?
    - n_features is large
    - Many uninformative features
  ]
]

#slide(title: "Many features, few observations: illustration in 1D")[
  #side-by-side(
    [
      #only(1)[
        #figure(image("img/pyfigures/linreg_noreg_0_nogrey.svg", width: 80%))
      ]
      #only(2)[
        #figure(image("img/pyfigures/linreg_noreg_0.svg", width: 80%))
      ]
    ],
    [
      - Few observations with respect to the number of features.
      - Fit a linear model without regularization.
      #uncover(2)[- Linear model can overfit if data is noisy.]
    ],
  )
]

#slide(title: "Many features, few observations: illustration in 1D")[
  = Sampling different training sets
  #grid(
    columns: (auto, auto, auto),
    gutter: 3pt,
    image("img/pyfigures/linreg_noreg_0.svg", width: 80%),
    image("img/pyfigures/linreg_noreg_1.svg", width: 80%),
    image("img/pyfigures/linreg_noreg_2.svg", width: 80%),

    image("img/pyfigures/linreg_noreg_3.svg", width: 80%),
    image("img/pyfigures/linreg_noreg_4.svg", width: 80%),
    image("img/pyfigures/linreg_noreg_5.svg", width: 80%),
  )
]

#slide(title: "Bias variance trade-off with Lasso")[
  #side-by-side(
    [
      #figure(image("img/pyfigures/linreg_noreg_0.svg", width: 80%))
      Linear regression (no regularization)

      High variance, no bias.
    ],
    [
      #figure(image("img/pyfigures/lasso_0_withreg.svg", width: 80%))
      Lasso (regularization): Shrink some coefficients of $beta$.

      Lower variance, but bias.
    ],
  )
]

#slide(title: "Bias variance trade-off with Lasso")[
  #grid(
    columns: (auto, auto, auto),
    gutter: 3pt,
    [
      #image("img/pyfigures/lasso_alpha_0.svg", width: 80%)
      Too much variance

      Not enough regularization
    ],
    [
      #image("img/pyfigures/lasso_alpha_10.svg", width: 80%)
      Best trade-off

    ],
    [
      #image("img/pyfigures/lasso_alpha_50.svg", width: 80%)
      Too much bias

      Too much regularization
    ],
  )
]

#slide(title: "Objective function of the Lasso")[
  The lasso puts a constrainst of amplitude $t$ on the $L_1$ norm of the coefficients:
  #eq($min_(beta) sum_i^(n)((y_i - beta^T x_i)^2) text("st.") sum_1^(p)|beta_j| <= t$)

  #uncover(2)[
    This is equivalent to the following optimization problem (using lagrangian multiplier):
    #eq($min_(beta) sum((y_i - beta^T x_i)^2) + alpha sum(|beta_j|)$)
  ]
  #uncover(3)[
    This penalty discourages large weights and can shrink certain weights to exactly _zero_ (not clear yet why).
  ]
]

#slide(title: "Why does Lasso shrink some coefficients to zero?")[
  #set align(center)
  #side-by-side(
    [
      #only(1)[
        #figure(image("img/pyfigures/lasso_intuition_inner.svg", width: 150%))
      ]
      #only(2)[
        #figure(image("img/pyfigures/lasso_intuition_middle.svg", width: 150%))
      ]
      #only(3)[
        #figure(image("img/pyfigures/lasso_intuition_outer.svg", width: 150%))
      ]
      #only(4)[
        #figure(image("img/pyfigures/lasso_intuition_penalty.svg", width: 150%))
      ]
    ],
    [
      #set align(left)
      - Plot the MSE of the model as a function of the coefficients.
      #uncover(2)[
        - The MSE surface is an ellispoid in $beta$.
      ]
      #uncover(4)[
        - The lasso objective function is a diamond.
      ]
    ],
  )
]

#slide(title: "Another regularized linear model: Ridge")[

  Ridge puts a constrainst of amplitude $t$ on the $L_2$ norm of the coefficients:
  #eq($min_(beta) sum_i^(n)((y_i - beta^T x_i)^2) text("st.") sum_1^(p)beta_j^2 <= t$)

  #uncover((2, 3))[
    This is equivalent to the following optimization problem (using lagrangian multiplier):
    #eq($min_(beta) sum((y_i - beta^T x_i)^2) + alpha sum(beta_j^2)$)
  ]
  #uncover(3)[
    This penalty shrinks the coefficients towards zero and each other.
  ]
]

#slide(title: "Importance of rescaling")[
  == Why rescale?

  - The penalty term in the Lasso and Ridge is sensitive to the scale of the features.

  - If the features are not on the same scale, the regularization will not be applied uniformly.

  - The coefficients of the model will be biased towards the features with the largest scale.

  == How to rescale?

  - Gaussian hypothesis? Standard scaling: $X = (X - text("mean")(X)) / (text("std")(X))$

  - Non Gaussian? MinMax scaling: $X = (X - min(X)) / (max(X) - min(X))$
]

#slide(title: "Importance of rescaling: illustration")[ ]

#slide(title: "Statistical model behind the Lasso")[

  = Many features, few observations

  #hyp_box(title: [Assumption 1: Linear model with high dimension])[

    $Y = X beta_0 + epsilon$, #h(1em) $epsilon tack.t.double X$ and $X in RR^(n times p)$ with $n << p$

  ]

  #hyp_box(title: [Assumption 2: (approximate) sparsity])[

    - The true $beta_0$ is sparse: ie. many coefficients are zero or very close to zero.
  ]
]

#slide(title: "Approximate sparsity: theoretical considerations")[
  #def_box(title: [Definition: Approximate sparsity])[
    The sorted absolute values of the coefficients decay quickly.

    $|beta|_((j)) < A j^(-a)$ #h(1em) $a> 1 / 2$

    for each j, where the constants a and A do not depend on the sample size n.
  ]
]

#slide(title: "Regularized models for classification")[
  TODO:
]


#slide(title: "How to choose lambda?")[
  == Theoretical garuantees

  - ⚠️ Assumptions are hard to verify.

  == In practice

  - Cross-validation: split the data in train and test sets, fit the model on the train set and evaluate on the test set.

  - We will look into that in more details in the next session.
]

#slide(title: [Short introduction to scikit-learn])[
  - url: https://github.com/strayMat/causal-ml-course/tree/main/notebooks
]

#new-section-slide("Python hands-on: Common pitfalls in the interpretation of coefficients of linear models")

#slide(title: [To your notebooks 🧑‍💻!])[
  - url: https://github.com/strayMat/causal-ml-course/tree/main/notebooks
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

  == Lasso

  - L1 penalty: sparsity
  - Feature selection
  - Unstable for correlated features

  == Ridge

  - L2 penalty: shrinkage
  - No feature selection
  - Stable for correlated features

]

#let bibliography = bibliography("biblio.bib", style: "apa")

#set align(left)
#set text(size: 20pt, style: "italic")
#slide[
  #bibliography
]
