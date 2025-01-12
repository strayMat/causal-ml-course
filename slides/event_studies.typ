// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.3.1": *
#import "@preview/embiggen:0.0.1": * // LaTeX-like delimiter sizing for Typst
#import "@preview/showybox:2.0.1": showybox
#import "@preview/codly:1.1.1": * // Code highlighting for Typst
#show: codly-init
#import "@preview/codly-languages:0.1.3": * // Code highlighting for Typst

#import themes.metropolis: *

#show: metropolis-theme//.with(footer: [Custom footer])
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

#let eq(body, size: 1.1em) = {
  set align(center)
  set text(size: size)
  body
}
// define custom tables
#set table(
  fill: (x, y) => if x == 0 or y == 0 {
    gray.lighten(40%)
  },
  align: right,
)
#show table.cell.where(x: 0): strong
#show table.cell.where(y: 0): strong

// assumption box
//
//
#let my_box(title, color, body) = {
  showybox(
    title-style: (
      color: black,
      weight: "light",
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
    align(left, body),
  )
}

#let hyp_box(title: "Assumption", body) = {
  my_box(title, rgb("#d39e57"), body)
}


#let def_box(title: "Definition", body) = {
  my_box(title, rgb("#57b4d3"), body)
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
    set text(size: 0.6em)
    show: pad.with(.5em)
    set align(bottom)
    text(fill: m-dark-teal.lighten(40%), m-footer.display())
    h(1fr)
    text(fill: m-dark-teal, logic.logical-slide.display())
  }

  set page(
    header: header,
    footer: footer,
    margin: (top: 3em, bottom: 0.5em),
    fill: m-extra-light-gray,
  )

  let content = {
    show: align.with(top)
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
  author: [Authors],
  title: "Machine Learning for econometrics",
  subtitle: "Event studies: Causal methods for pannel data",
  date: "February, 11th, 2025",
)


#new-section-slide("Motivation")

#slide(title: "Setup: event studies")[
  == Estimation of the effect of a treatment when data is:
  Aggregated: country-level data such as employment rate, GDP...
  #only(1)[
    #figure(image("img/event_studies/aggregation_units.svg", width: 50%))
  ]

  #only((2, 3, 4, 5))[
    Longitudinal: multiple time periods (or repeated cross-sections)...
  ]
  #only(2)[
    #figure(image("img/event_studies/multiple_time_periods.svg", width: 50%))
  ]

  #only((3, 4, 5))[
    With multiple aggregated units: countries, firms, geographical regions...
  ]
  #only(3)[
    #figure(
      image("img/event_studies/geographic_units.png", width: 40%),
      caption: [Figure from @degli2020can],
    )
  ]

  #only((4, 5))[
    Staggered adoption of the treatment: units adopt the policy/treatment at different times...
  ]
  #only(4)[

    #figure(image("img/event_studies/staggered_adoption.svg", width: 40%))
  ]

  #only(5)[
    This setup is known as
    #set align(center)
    == #alert[Panel data, event studies, longitudinal data, time-series data.]
  ]
]

#slide(title: "Examples of event studies")[

  === Archetypal questions

  - Did the new marketing campaign had an effect on the sales of a product?

  - Did the new tax policy had an effect on the consumption of a specific product?

  - Did the guidelines on the prescription of a specific drug had an effect on the practices?

  #pause
  === Modern examples

  - What is the effect of the extension of Medicaid on mortality? @miller2019medicaid

  - What is the effect of Europe’s protected area policies (_Natura 2000_) on vegetation cover and on economic activity? @grupp2023evaluation

  - Which policies achieved major carbon emission reductions? @stechemesser2024climate
]

#slide(title: "Setup: event studies are quasi-experiment")[

  #def_box(title: "Quasi-experiment")[
    A situation where the treatment is not randomly assigned by the researcher but by nature or society. #linebreak()
    It should introduce _some_ randomness in the treatment assignment: enforcing treatment exogeneity, ie. ignorability (ie. unconfoundedness).
  ]

  #pause
  == Other quasi-experiment designs

  - #alert[Instrumental variables:] a variable that is correlated with the treatment but not with the outcome.

  - #alert[Regression discontinuity design:] the treatment is assigned based on a threshold of a continuous variable.
]



#slide(title: "Table of contents")[
  #metropolis-outline
]
#new-section-slide("Reminder on difference-in-differences")

#slide(title: "Difference-in-differences")[

  == History

  - First documented example (though not formalized): John Snow showing how cholera spread through the water in London @snow1855mode #footnote[#text(size: 15pt)[Good description: #link("https://mixtape.scunning.com/09-difference_in_differences#john-snows-cholera-hypothesis")]]

  - Modern usage introduced formally by @ashenfelter1978estimating, applied to labor economics

  == Idea
  - Contrast the temporal effect of the treated unit with the control unit temporal effect:

  - The difference between the two differences is the treatment effect
]

#slide(title: "Difference-in-differences framework")[
  #only(1)[
    == Two period of times: t=1, t=2
    #figure(image("img/pyfigures/did_setup.svg", width: 50%))
  ]
  #only((2, 3))[
    === Potential outcomes: $Y_t (d)$ where $d={0,1}$ is the treatment at period 2

    #figure(image("img/pyfigures/did_t1_factual.svg", width: 50%))
  ]
  #only(3)[
    ⚠️ $EE[Y_1(1)] = EE[Y_1 (1) |D=1] PP(D=1) +  EE[Y_1 (1) |D=0] PP(D=0)$ #linebreak() but we only observe $EE[Y_1 (1) |D=1]$
  ]
  #only((4, 5))[
    === Our target is the average treatment effect on the treated (ATT)

    #figure(image("img/pyfigures/did_factual.svg", width: 50%))
  ]

  #only(4)[$tau_("ATT") = EE[Y_2 (1)| D = 1] - EE [Y_2(0)| D = 1]$]

  #only(5)[$tau_("ATT") = underbrace([Y_2 (1)| D = 1], #c_treated("treated outcome for t=2")) - underbrace(EE [Y_2(0)| D = 1], "unobserved counterfactual")$]

  #only(6)[
    === First assumption, parallel trends

    $EE[Y_2(0) - Y_1(0) | D = 1] = EE[Y_2(0) - Y_1(0) | D = 0]$
    #figure(image("img/pyfigures/did_parallel_trends.svg", width: 50%))
  ]
  #only(7)[
    === First assumption, parallel trends // #only(7)[#footnote[#text("⚠️ Strong assumption ! We will come back to it later.", size: 15pt)]
    $underbrace([Y_2(0) - Y_1(0) | D = 1], #c_treated("Trend(1)")) = underbrace(EE[Y_2(0) - Y_1(0) | D = 0], #c_control("Trend(0)"))$
    #figure(image("img/pyfigures/did_parallel_trends_w_coefs.svg", width: 50%))
  ]

  #only(8)[
    === First assumption, parallel trends

    $EE[Y_2(0) | D = 1]= EE[Y_1(0) | D = 1] + EE[Y_2(0) - Y_1(0) | D = 0]$
    #figure(image("img/pyfigures/did_parallel_trends_w_coefs.svg", width: 50%))
  ]
  #only(8)[
    === First assumption, parallel trends

    $EE[Y_2(0) | D = 1]= underbrace([Y_1(0) | D = 1], "unobserved counterfactual") + EE[Y_2(0) - Y_1(0) | D = 0]$
    #figure(image("img/pyfigures/did_parallel_trends_w_coefs.svg", width: 50%))
  ]

  #only(9)[
    === Second assumption, no anticipation of the treatment

    $E[Y_1(1)|D=1]=E[Y_1(0)|D=1]$

    #figure(image("img/pyfigures/did_no_anticipation.svg", width: 50%))
  ]
]

#slide(title: "Difference-in-differences framework: identification of ATT")[

  $tau_("ATT") &= EE[Y_2(1)| D = 1] - EE [Y_2(0)| D = 1]\
    &= underbrace(EE[Y_2(1)| D = 1] - EE[Y_1(0)|D=1], #c_treated("Factual Trend(1)")) - underbrace(EE[Y_2(0)|D=0] - EE[Y_1(0)|D=0], #c_control("Trend(0)"))$
  #figure(image("img/pyfigures/did_att.svg", width: 50%))
]

#slide(title: "Estimation: link with two way fixed effect (TWFE)")[

  #eq[$Y = alpha + gamma D + lambda bb(1) (t=2) + tau_("ATT") D bb(1) (t=2)$]

  #figure(image("img/pyfigures/did_twfe.svg", width: 50%))

  ⚠️ Mechanic link: works only under parallel trends and no anticipation assumptions.
]

//#slide(title: "Failure of the no-anticipation assumption")[]


#slide(title: "Failure of the parallel trend assumption")[
  #only(1)[
    == Seems like the treatment decreases the outcome!
    #figure(image("img/pyfigures/did_non_parallel_trends_last_periods.svg", width: 90%))
  ]

  #only(2)[
    == Oups...
    #figure(image("img/pyfigures/did_non_parallel_trends_all_periods.svg", width: 90%))
  ]
]


#slide(title: "DID estimator for more than two time units")[
  === Target estimand: sample average treatment effect on the treated (SATT)

  #eq($tau_(text("SATT")) = 1 / (|{i \: D_i=1}|) sum_(i:D_i=1) 1 / (T-H) sum_(t=H+1)^T Y_(i t)(1) - Y_(i t)(0)$)

  === DID estimator

  #eq($hat(tau_(text("DID"))) = 1 / (|{i \: D_i=1}|) sum_(i:D_i=1) [1 / (T-H) sum_(t=H+1)^T Y_(i t) - 1 / H sum_(t=1)^H Y_(i t)] - 1 / (|{i \: D_i=0}|) sum_(i:D_i=0) [1 / (T-H) sum_(t=H+1)^T Y_(i t) - 1 / H sum_(t=1)^H Y_(i t)]$)

  #hyp_box[
    === No anticipation of the treatment: $Y_(i t)(0) = Y_(i t)(1) forall t = 1,..., H.$

    === Parallel trend: $EE [Y_(i t)(0, infinity) - Y_(i 1)(0, infinity)] = beta_t, t = 2,..., T.$
  ]
  See @wager2024causal for a clear proof of consistancy.
]


#slide(title: "DID: Take-away")[
  == Pros
  - Extremely common in economics and quite simple to implement.
  - Can be extended to @wager2024causal
    - more than two time periods: exact same formulation
    - staggered adoption of the treatment: a bit more complex

  == Cons
  - Very strong assumptions: parallel trends and no anticipation.
  - Does not account for heterogeneity of treatment effect over time @de2020two.

  #pause

  == Can we do better: ie. robust to the parallel trend assumption?
]
//# TODO?
//#slide(title:"Inference for DID")[]

#new-section-slide("Synthetic controls")

#slide(title: "Synthetic controls")[

  === References

  Introduced by @abadie2003economic and @abadie2010synthetic.

  Quick introduction in @bonander2021synthetic, technical overiew in @abadie2021using,

  #pause

  #quote(attribution: [@athey2017state], block: true)[
    The most important innovation in the policy evaluation literature in the last few years
  ]

  #pause
  === Idea
  Find a weighted average of controls that predicts well the treated unit outcome before treatment.

  == Example
  What is the effect of tobacco tax on cigarettes sales? @abadie2010synthetic
]

#slide(title: "Examples of application of synthetic controls to epidemiology")[
  - What is the effect of taxes on sugar-based product consumption @puig2021impact

]

#slide(title: [Synthetic control example: California's Proposition 99 @abadie2010synthetic])[

  == Context

  1988: 25-cent tax per pack of cigarettes, ban of on cigarette vending machines in public areas accessible by juveniles, and a ban on the individual sale of single cigarettes.

  #pause

  == Setup

  === Outcome, $Y_(j, t)$: cigarette sales per capita
  #pause

  === Treated unit, $j=1$: California as from 1988
  #pause

  === Control units, $j in {2,..J}$: 39 other US states without similar policies
  #pause

  === Time period: $t in {1,..T} = {1970, ..2000}$ and treatment time $T_0 = 1988$

  #pause

  == Covariates $X_(j, t)$: cigarette price, previous cigarette sales.
]

#slide(title: [Synthetic control example: plot the data])[
  #figure(image("img/pyfigures/scm_california_vs_other_states.svg", width: 70%))


  #pause
  😯 Decrease in cigarette sales in California.

  #pause
  🤔 Decrease began before the treatment and occured also for other states.
]

#slide(title: [Synthetic control example: plot the data])[
  #figure(image("img/pyfigures/scm_california_and_other_states.svg", width: 70%))
  #pause
  💡 Force parallel trends: Find a weighted average of other states that predicts well the pre-treatment trend of California (before $T_0=1988$).
]

#slide(title: "Synthetic control as weighted average of control outcomes")[

  #side-by-side()[
    Build a predictor for #c_treated($Y_(1, t)$) (California):

    $#c_treated[$hat(Y)_(1, t)$] = sum_(j=2)^(n_0 + 1) hat(w)_j #c_control[$Y_(j, t)$]$

    #only((2, 3, 4))[
      🤔How to choose the weights?
    ]

    #only((2, 3, 4))[
      Minimize some distance between the treated and the controls.
    ]


    #only(4)[
      🤓 This is called a balancing estimator: kind of Inverse Probability Weighting @wager2024causal[chapter 7]
    ]
  ][
    #figure(image("img/event_studies/scm_weighted_average.png", width: 100%))
  ]

]

#slide(title: "Synthetic controls: minimization problem")[

  === Characteristics

  Pre-treatment characteristics concatenate pre-treatment outcomes and other pre-treatment predictors $Z_1$ eg. cigarette prices:

  #c_treated[$X_("treat") = X_1 = vec(Y_(1, 1), Y_(1,2), .., Y_(1, T_0), Z_1)$] $in R^(p times 1)$


  #pause
  Let the control pre-treatment characteristics be: #c_control[$X_("control") = (X_2, .., X_(n_0 + 1))$] $in R^(p times n_0)$

  === Minimization problem #only(5)[with constraints]

  #set align(center)
  #only((3, 4))[
    $w^(*) &= "argmin"_(w) ||X_("treat") - X_("control") w||_V^2$
  ]

  #only(4)[
    where $||X||_V = sqrt(X^T V X) " with " V in "diag"(R^p)$

    This gives more importance to some features than others.
  ]
  #only(5)[
    $w^(*) &= "argmin"_(w) ||X_("treat") - X_("control") w||_V^2\
      &s.t. space w_j >= 0, \
      &sum_(j=2)^(n_0 + 1) w_j = 1$
  ]
]


#slide(title: "Synthetic controls: Why choose positive weights summing to one?")[
  #pause
  == This is called interpolation (vs extrapolation)

  #figure(image("img/event_studies/extrapolation.png", width: 70%))

  #pause
  == Interpolation enforces regularization, thus limits overfitting

  Same kind of regularization than L1 norm in Lasso: forces some coefficient to be zero.

  // (both are #link("https://en.wikipedia.org/wiki/Convex_optimization", [_optimization with constraints on a simplex_])).

]

#slide(title: "Synthetic controls: Extrapolation failure with unconstrained weight")[
  #set align(horizon)
  #side-by-side(columns: (2fr, 2fr))[
    $p=2 T_0$ covariates:

    $X_j= vec(Y_(j, 1), ..,Y_(j, T_0), Z_(j, 1), .., Z_(j, T_0))^T in R^(2T_0)$

    Y cigarette sales, Z cigarette prices.

    #pause
    Model: $underbrace(X_("treat"), p times 1)  tilde underbrace(X_("control"), p times n_0) underbrace(w, n_0)$

    #only(2)[#alert("-> simple linear regression estimated by OLS")]

    #pause
    Prediction: $hat(Y)_("synth") = vec(Y_(t, j))_(t=1..T#linebreak()j=2..n_0+1) w$
  ][
    #pause
    #figure(image("img/pyfigures/scm_california_vs_synth_lr.svg", width: 100%))
  ]
  #only(5)[=== 😭 Overfitting]
]


#slide(title: [Synthetic controls: How to choose the predictor weights $V$?])[

  1. Don't choose: set $V = I_(p)$, ie. $||X||_V = ||X||_2$.

  #pause
  2. Rescale by the variance of the predictors: #linebreak() $V = "diag"("var"(Y_(j, 1))^(-1), .., "var"(Y_(j, T_0))^(-1), "var"(Z_(j, 1))^(-1), .., "var"(Z_(j, T_0))^(-1))$.

  #pause
  3. Minimize the pre-treatment mean squared prediction error (MSPE) of the treated unit:

  $"MSPE"(V) &= sum_(t=1)^(T_0) [Y_(1, t) - sum_(j=2)^(n_0+1) w_j^*(V) Y_(j, t)]^2 \
    &= || vec(Y_(1, t))_(t=1..T_0) - vec(Y_(j, t))^T_(j=2..n_0+1#linebreak()t=1..T_0) hat(w) ||_(2)^(2)$

  This solution is solved by running two optimization problems:
  - #alert[Inner loop] solving $w^*(V)= "argmin"_(w) ||X_("treat") - X_("control") w||_V^2$

  - #alert[Outer loop] solving $V^*= "argmin"_(V) "MSPE"(V)$
]
//TODO: cross validation for the choice of V

#slide(title: "Synthetic controls: estimation without the outer optimization problem")[
  #side-by-side(columns: (1.5fr, 2fr))[

    Same coviarates: $X_j= vec(Y_(j, 1), ..,Y_(j, T_0), Z_(j, 1), .., Z_(j, T_0))^T$

    Y cigarette sales, Z cigarette prices.


    SCM minization with $V = I_(p)$, hence, $||X||_V = ||X||_2$.
    #v(1em)

    $w^(*) &= "argmin"_(w) ||X_("treat") - X_("control") w||_2^2\
      &s.t. space w_j >= 0, \
      &sum_(j=2)^(n_0 + 1) w_j = 1$

  ][
    #pause
    #figure(image("img/pyfigures/scm_california_vs_synth_wo_v.svg", width: 100%))
  ]

]


#slide(title: "Synthetic controls: estimation with the outer optimization problem")[
  #figure(image("img/pyfigures/scm_california_vs_synth_pysyncon.svg", width: 70%))
]

#slide(title: "Synthetic controls: inference")[

  === Variability does not come from the variability of the outcomes

  Indeed, aggregates are often not very noisy (once deseasonalized)...

  #pause
  === ... but from the variability of the chosen control units

  Treatment assignment introduces more noise than outcome variability.

  #pause

  @abadie2010synthetic introduced the placebo test to assess the variability of the synthetic control.

]


#slide(title: "Synthetic controls: inference with Placebo tests")[
  === Idea of Fisher’s Exact tests

  - Permute the treated and control exhaustively.

  - For each unit, we pretend it is the treated while the others are the control: we call it a placebo

  - Compute the synthetic control for each placebo: it should be close to zero.

]


#slide(title: "Synthetic controls: inference with Placebo tests, example")[
  #only((1, 2))[
    === Placebo estimation for all 38 control states

    #figure(image("img/pyfigures/scm_placebo_test.svg", width: 70%))
  ]
  #only(2)[
    - More variance after the treatment for California than before.
    - Some states have pre-treatment trends which are hard to predict.
  ]
  #only(3)[
    === Placebo estimation for 34 control states with "good" pre-treatment fit

    #figure(image("img/pyfigures/scm_placebo_test_wo_outliers.svg", width: 70%))

    I removed the states above the 90 percentiles of the distribution of the pre-treatment fit.

  ]
]

#slide(title: "Synthetic controls: inference with Placebo tests, example")[
  #side-by-side()[
    === California absolute cumulative effect

    $hat(tau)_("scm, california")=-17.00$

    #only(2)[
      === Get a p-value
      #h(1em)
      $"PV" &= 1 / (n_0) sum_(j=2)^(n_0) bb(1) big("(") |hat(tau)_("scm, california")| > |hat(tau)_("scm", j)| big(")")\ &= 0.029$
    ]

  ][
    #figure(image("img/pyfigures/scm_placebo_test_distribution.svg", width: 100%))
  ]
]

#slide(title: "Synthetic controls: inference with conformal prediction")[

]

#slide(title: "Synthetic controls: Take-away")[
  == Pros
  - More convincing for parallel trends assumption.
  - Simple for multiple time periods.
  - Gives confidence intervals.

  == Cons
  - Requires many control units to yield good pre-treatment fits.
  - Might be prone to overfitting during the pre-treatment period.
  - Still requires a strong assumption: the weights should also balance the post-treatment unexposed outcomes. See @arkhangelsky2021synthetic for discussions.
  - Still requires the no-anticipation assumption.
]


#new-section-slide("Conditional difference-in-differences")


#new-section-slide("Time-series modelisation: methods without a control group")

#slide(title: "Interrupted Time Series: intuition")[
  == Setup

  - One #c_treated[treated unit], no #c_control[control unit].
  - Multiple time periods.
  - Sometimes, predictors are availables: there are called exogeneous covariates.

  #pause
  == Intuition

  - Model the pre-treatment trend: $Y_t(1) "for" t<T_0$
  - Predict post-treatment trend as the control: $#c_control[$hat(Y_t)$(0)] "for" t>T_0 $

  - Obtain treatment effect by taking the difference between observed and predicted post-treatment observations: $#c_treated[$Y_(t)(1)$] - #c_control[$hat(Y_t)$(0)]$
]

#slide(title: [Interrupted Time Series: illustration from @schaffer2021interrupted])[

  #figure(image("img/event_studies/its_illustration.png", width: 65%))

  #set text(size: 18pt)
  $Y_t$: Dispensations of quetiapine, an anti-psychotic medicine.

  Treatment: Restriction of the conditions under which quetiapine could be subsidised.

]

#slide(title: "Modelization of a time-series")[

  == Tools

  - ARIMA models: AutoRegressive Integrated Moving Average

  #my_box(
    "Motivation of ARIMA",
    rgb("#57b4d3"),
    [
      Take into account:
      - the structure of autodependance between observation,
      - linear trends,
      - seasonality.
    ],
  )


  A good reference for ARIMA: #link("https://otexts.com/fpp3/","Forecasting: Principles and Practice, chapter 8")

]

#slide(title: [ARIMA are State Space Models (SSM) #text("says the machine learning community",size: 18pt)])[
  == Why showing this formulation ?

  - I better understand ARIMA formulated as state space models.

  - SSM are more general than ARIMA models.

  - ARIMA are (almost always) fitted with SSM optimization algorithms.

  == What is a state space model?

]

#slide(title: "A word on model families for ITS")[
  We saw ARIMA models and the more general class of state space models.

  However, we could any model that we want to fit the pre-treatment trend !

  #pause
  - #link("https://facebook.github.io/prophet/", [Facebook prophet model @taylor2018forecasting]) uses Generalized Additive Models (GAM).

  #pause
  - Any sklearn estimator could do the trick: Linear regression, Random Forest, Gradient Boosting...

  #pause
  ⚠️ You should pay attention to appropriate train/test split when cross-validating a time-series model not to use the future to predict the past.

  Relevant remark for all time series models (even ARIMA or state space models).
]

#slide(title: "Cross-validation for time-series models")[

  ```python
  from sklearn.model_selection import TimeSeriesSplit
  ```
  #figure(image("img/event_studies/time_series_cv.png", width: 70%))

  This avoids to use the future to predict the past.
]

#slide(title: "Take-away on ITS")[

  == Pros

  - Suitable when no control unit is available. The pre-treatment trend is the control.

  - Handles multiple time periods.

  - A lot of software available (eg. ARIMA models).

  - Simple: few parameters to tune.

  == Cons

  - Prone to bias by other events happening around the treatment time and impacting the outcome trend.

  - Prone to overfitting of the pre-treatment trend.
]


#slide(title: "An attempt to map event study methods")[
  #set text(size: 15pt)

  #table(
    columns: 6,
    align: (left, left, center, center, center),
    table.header("Methods", "Characteristics", "Hypotheses", "Community", "Introduction", "Good reference"),
    "DID/TWFE",
    [Treated/control units, few time periods, no predictors],
    [Parallel trends, no anticipation, prone to overfitting],
    [Economics],
    [#link(
        "https://matheusfacure.github.io/python-causality-handbook/13-Difference-in-Differences.html",
        "Causal Inference for the Brave and True, chapter 13",
      )],
    [@arkhangelsky2024causal],

    "ARIMA, ITS",
    [No controls, no/few predictors, seasonality],
    [Stationnarity , no anticipation, prone to overfitting],
    [Epidemiology, Economics],
    [#link("https://otexts.com/fpp3/arima.html", "Forecasting: Principles and Practice")],
    [@schaffer2021interrupted],

    "State space models",
    [Multiple time periods, control units or predictors, generalization of ARIMA],
    [Contional ignorability on predictors, goodness of fit pre-treatment],
    [Machine learning, bayesian methods],
    [#link(
        "repository.cinec.edu/bitstream/cinec20/1109/1/2016_Book_IntroductionToTimeSeriesAndFor.pdf#page=259",
        [@brockwell2016introduction[chapter 9]],
      )],
    [@murphy2022probabilistic[chapter 18]],

    "Synthetic control",
    [Treated/control units, multiple time periods],
    [Conditional parallel trend on controls, goodness of fit pre-treatment],
    [Economics],
    [#link(
        "https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html",
        "Causal Inference for the Brave and True",
      )],
    [@abadie2021using],
  )

]

#slide(title: "A summary on R packages for event studies")[

  #table(
    columns: 5,
    align: (left, left, center, center, center),
    table.header([Package name], "Methods", "Predictors", "Control units", "Multiple time periods"),
    [#link("https://cran.r-project.org/web/packages/did/index.html", "did")],
    "Difference-in-differences",
    [❌],
    [❌],
    [❌],

    [#link("https://pkg.robjhyndman.com/forecast/reference/Arima.html", "forecast")], "ARIMA, ITS", [✅], [❌], [✅],
    [#link("https://cran.r-project.org/web/packages/Synth/index.html", "Synth")], "Synthetic control", [❌], [✅], [✅],
    [#link("https://github.com/google/CausalImpact", "Causal impact")], "Bayesian state space models", [✅], [❌], [✅],
  )
]


#slide(title: "A summary on Python packages for event studies")[
  #set text(size: 18pt)
  #table(
    columns: 5,
    align: (left, left, center, center, center),
    table.header([Package name], [Methods], [Predictors], [Control units], [Multiple time periods]),
    [#link("https://www.statsmodels.org/stable/regression.html", "statsmodels.OLS")],
    "Difference-in-differences, TWFE",
    [❌],
    [❌],
    [❌],

    [#link("https://www.statsmodels.org/stable/examples/index.html#state-space-models", "statsmodels")],
    "ARIMA(X), ITS, bayesian state space models",
    [✅],
    [❌],
    [✅],

    [#link("https://alkaline-ml.com/pmdarima/index.html", "pmdarima")], "ARIMA(X), ITS", [✅], [❌], [✅],
    [#link("https://github.com/OscarEngelbrektson/SyntheticControlMethods", "SyntheticControlMethods")],
    "Synthetic control",
    [❌],
    [✅],
    [✅],

    [#link("https://github.com/sdfordham/pysyncon", "pysyncon")], "Synthetic control", [❌], [✅], [✅],
    [#link("https://github.com/jamalsenouci/causalimpact/", "causalimpact (pymc implementation)") ],
    "Bayesian state space models",
    [✅],
    [❌],
    [✅],

    [#link("https://github.com/tcassou/causal_impact/tree/master", "causal-impact (statsmodels implementation)")],
    "Bayesian state space models",
    [✅],
    [❌],
    [✅],
  )
]

#new-section-slide("Python hands-on")

#slide(title: [To your notebooks 🧑‍💻!])[
  - url: https://github.com/strayMat/causal-ml-course/tree/main/notebooks
]


#let bibliography = bibliography("biblio.bib", style: "apa")

#set align(left)
#set text(size: 20pt, style: "italic")

#slide(title: "Bibliography")[
  #bibliography
]
