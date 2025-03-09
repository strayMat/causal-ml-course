// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.4.0": *
#import "@preview/embiggen:0.0.1": * // LaTeX-like delimiter sizing for Typst
#import "@preview/showybox:2.0.4": showybox
#import "@preview/codly:1.1.1": * // Code highlighting for Typst
#show: codly-init
#import "@preview/codly-languages:0.1.3": * // Code highlighting for Typst
#import "@preview/fletcher:0.5.6" as fletcher: diagram, node, edge // for dags
#import "@preview/metropolis-polylux:0.1.0" as metropolis
#import metropolis: new-section, focus

#show: metropolis.setup


#set text(size: 22pt)
#set par(justify: true)
#set figure(numbering: none)
#show math.equation: set text(font: "Latin Modern Math")
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

#show link: set text(fill: rgb("#3788d3"))

#let bibliography = bibliography("biblio.bib", style: "apa")


#slide[
  #set page(header: none, footer: none, margin: 3em)

  #text(size: 1.3em)[
    *Machine Learning for econometrics*
  ]

  Event studies: Causal methods for pannel data

  #metropolis.divider

  #set text(size: .8em, weight: "light")
  Matthieu Doutreligne
  March, 11th, 2025
]

#let alert = body => {
  set text(fill: orange)
  body
}


#new-section["Motivation"]

#slide[
  = Setup
  == Estimation of the effect of a treatment when data is:
  - Aggregated: country-level data such as employment rate, GDP...
  #only(1)[
    #figure(image("img/event_studies/aggregation_units.svg", width: 50%))
  ]

  #only((2, 3, 4, 5))[
    - Longitudinal: multiple time periods (or repeated cross-sections)...
  ]
  #only(2)[
    #figure(image("img/event_studies/multiple_time_periods.svg", width: 50%))
  ]

  #only((3, 4, 5))[
    - With multiple aggregated units: countries, firms, geographical regions...
  ]
  #only(3)[
    #figure(
      image("img/event_studies/geographic_units.png", width: 40%),
      caption: [Figure from @degli2020can],
    )
  ]

  #only((4, 5))[
    - Staggered adoption of the treatment: units adopt the policy/treatment at different times...
  ]
  #only(4)[
    #figure(image("img/event_studies/staggered_adoption.svg", width: 37%))
  ]

  #only(5)[
    This setup is known as:
    #set align(center)
    == #alert[Panel data, event studies, longitudinal data, time-series data.]
  ]
]

#slide[
  = Examples of event studies
  === Archetypal questions

  - Did the new marketing campaign had an effect on the sales of a product?

  - Did the new tax policy had an effect on the consumption of a specific product?

  - Did the guidelines on the prescription of a specific drug had an effect on the practices?

  #show: later
  === Modern examples

  - What is the effect of the extension of Medicaid on mortality? @miller2019medicaid

  - What is the effect of Europe’s protected area policies (_Natura 2000_) on vegetation cover and on economic activity? @grupp2023evaluation

  - Which policies achieved major carbon emission reductions? @stechemesser2024climate
]

#slide[
  = Setup: event studies are quasi-experiment
  #def_box(title: "Quasi-experiment")[
    A situation where the treatment is not randomly assigned by the researcher but by nature or society. #linebreak()
    It should introduce _some_ randomness in the treatment assignment: enforcing treatment exogeneity, i.e. ignorability (i.e. unconfoundedness).
  ]

  #show: later
  == Other quasi-experiment designs

  - #alert[Instrumental variables:] a variable that is correlated with the treatment but not with the outcome.

  - #alert[Regression discontinuity design:] the treatment is assigned based on a threshold of a continuous variable.
]


#new-section["Reminder on difference-in-differences"]

#slide[
  = Difference-in-differences
  == History

  - First documented example (though not formalized): John Snow showing how cholera spread through the water in London @snow1855mode #footnote[#text(size: 15pt)[Good description: #link("https://mixtape.scunning.com/09-difference_in_differences#john-snows-cholera-hypothesis")]]

  - Modern usage introduced formally by @ashenfelter1978estimating, applied to labor economics

  #show: later
  == Idea
  - Contrast the temporal effect of the treated unit with the control unit temporal effect.

  - The difference between the two differences is the treatment effect.
]

#slide[
  = Difference-in-differences

  #only(1)[
    == Two period of times: t=1, t=2
    #figure(image("img/pyfigures/did_setup.svg", width: 50%))
  ]
  #only((2, 3))[
    === Potential outcomes: $Y_t (d)$ where $d={0,1}$ is the treatment at period 2

    #figure(image("img/pyfigures/did_t1_factual.svg", width: 50%))
  ]
  #only(3)[
    ⚠️ $EE[Y_1(1)] = underbrace([EE[Y_1 (1) |D=0]], "counterfactural") PP(D=0) + underbrace([Y_1 (1) |D=1], "observed") PP(D=1)$
  ]
  #only((4, 5))[
    === Our target is the average treatment effect on the treated (ATT)

    #figure(image("img/pyfigures/did_factual.svg", width: 50%))
  ]

  #only(4)[$tau_("ATT") = EE[Y_2 (1)| D = 1] - EE [Y_2(0)| D = 1]$]

  #only(5)[$tau_("ATT") = underbrace([Y_2 (1)| D = 1], #c_treated("treated outcome for t=2")) - underbrace(EE [Y_2(0)| D = 1], "unobserved counterfactual")$]

  #only(6)[
    = Difference-in-differences

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
    = Difference-in-differences

    === First assumption, parallel trends

    $EE[Y_2(0) | D = 1]= underbrace(EE[Y_1(0) | D = 1], "unobserved counterfactual") + EE[Y_2(0) - Y_1(0) | D = 0]$
    #figure(image("img/pyfigures/did_parallel_trends_w_coefs.svg", width: 50%))
  ]

  #only(9)[
    === Second assumption, no anticipation of the treatment

    $EE[Y_1(1)|D=1]=EE[Y_1(0)|D=1]$

    #figure(image("img/pyfigures/did_no_anticipation.svg", width: 50%))
  ]
]

#slide[
  = Identification of ATT

  $tau_("ATT") &= EE[Y_2(1)| D = 1] - EE [Y_2(0)| D = 1]\
    &= underbrace(EE[Y_2(1)| D = 1] - EE[Y_1(0)|D=1], #c_treated("Factual Trend(1)")) - underbrace(EE[Y_2(0)|D=0] - EE[Y_1(0)|D=0], #c_control("Trend(0)"))$
  #figure(image("img/pyfigures/did_att.svg", width: 50%))
]

#slide[
  = Estimation: link with two way fixed effect (TWFE)
  #eq[$Y = alpha + gamma D + lambda bb(1) (t=2) + tau_("ATT") D bb(1) (t=2)$]

  #figure(image("img/pyfigures/did_twfe.svg", width: 50%))

  ⚠️ Mechanic link: works only under parallel trends and no anticipation assumptions.
]

#slide[
  = Failure of the parallel trend assumption
  #only(1)[
    == Seems like the treatment decreases the outcome!
    #figure(image("img/pyfigures/did_non_parallel_trends_last_periods.svg", width: 90%))
  ]

  #only(2)[
    == Oups...
    #figure(image("img/pyfigures/did_non_parallel_trends_all_periods.svg", width: 90%))
  ]
]


#slide[
  = DID estimator for more than two time units

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


#slide[
  = DID: Take-away
  == Pros
  - Extremely common in economics and quite simple to implement.
  - Can be extended to @wager2024causal
    - more than two time periods: exact same formulation
    - staggered adoption of the treatment: a bit more complex

  #show: later
  == Cons
  - Strong assumptions: parallel trends and no anticipation.
  - Does not account for heterogeneity of treatment effect over time @de2020two.

  #show: later

  == Can we do better: i.e. robust to the parallel trend assumption?
]

#new-section["Synthetic controls"]

#slide[
  = Synthetic Control Methods (SCM)

  Introduced by @abadie2003economic and @abadie2010synthetic.

  Quick introduction in @bonander2021synthetic, technical overiew in @abadie2021using,

  #show: later

  #quote(attribution: [@athey2017state], block: true)[
    The most important innovation in the policy evaluation literature in the last few years
  ]

  #show: later
  === Idea
  Find a weighted average of controls that predicts well the treated unit outcome before treatment.

  == Example
  What is the effect of tobacco tax on cigarettes sales? @abadie2010synthetic
]

#slide[
  = Examples of application of synthetic controls to epidemiology

  - Literature review of the usage of SCM in healthcare (up to 2016): @bouttell2018synthetic

  == Some use cases

  - What is the effect of UK pay-for-performance program in primary care on mortality? @ryan2016long

  - What is the effect of soda taxes on sugar-based product consumption? @puig2021impact

  - What is the effect of Ohio vaccine lottery on covid-19 vaccination? @brehm2022ohio

  - What is the effect of wildfire storm on respiratory hospitalizations? @sheridan2022using
]

#slide[
  = Synthetic control example: California's Proposition 99 @abadie2010synthetic

  == Context

  1988: 25-cent tax per pack of cigarettes, ban of on cigarette vending machines in public areas accessible by juveniles, and a ban on the individual sale of single cigarettes.

  #show: later

  == Setup

  === Outcome, $Y_(j, t)$: cigarette sales per capita
  #show: later

  === Treated unit, $j=1$: California as from 1988
  #show: later

  === Control units, $j in {2,..J}$: 39 other US states without similar policies
  #show: later

  === Time period: $t in {1,..T} = {1970, ..2000}$ and treatment time $T_0 = 1988$

  #show: later

  == Covariates $X_(j, t)$: cigarette price, previous cigarette sales.
]

#slide[
  = Synthetic control example: plot the data
  #figure(image("img/pyfigures/scm_california_vs_other_states.svg", width: 70%))


  #show: later
  😯 Decrease in cigarette sales in California.

  #show: later
  🤔 Decrease began before the treatment and occured also for other states.
]

#slide[
  = Synthetic control example: plot the data
  #figure(image("img/pyfigures/scm_california_and_other_states.svg", width: 70%))
  #show: later
  💡 Force parallel trends: Find a weighted average of other states that predicts well the pre-treatment trend of California (before $T_0=1988$).
]

#slide[
  = Synthetic control as weighted average of control outcomes

  #toolbox.side-by-side()[
    Build a predictor for #c_treated($Y_(1, t)$) (California):

    $#c_treated[$hat(Y)_(1, t)$] = sum_(j=2)^(n_0 + 1) hat(w)_j #c_control[$Y_(j, t)$]$

    #only((2, 3, 4))[
      🤔How to choose the weights?
    ]

    #only((2, 3, 4))[
      Minimize some distance between the treated and the controls.
    ]

    #only(4)[
      🤓 This is called a balancing estimator: kind of Inverse Probability Weighting.

      Cf. @wager2024causal[chapter 7] for details on balancing estimators.
    ]
  ][
    #figure(image("img/event_studies/scm_weighted_average.png", width: 100%))
  ]

]

#slide[
  = Synthetic controls: minimization problem
  === Characteristics

  Pre-treatment characteristics concatenate pre-treatment outcomes and other pre-treatment predictors $Z_1$ eg. cigarette prices:

  #toolbox.side-by-side()[
    #c_treated[$X_("treat") = X_1 = vec(Y_(1, 1), Y_(1,2), ..., Y_(1, T_0), Z_1)$] $in R^(p times 1)$
  ][
    #c_control[$X_("control") = (X_2, .., X_(n_0 + 1))$] $in R^(p times n_0)$
  ]

  === Minimization problem #only(5)[with constraints]

  #set align(center)
  #only((3, 4))[
    $w^(*) &= "argmin"_(w) ||X_("treat") - X_("control") w||_V^2$
  ]

  #only(4)[
    where $||X||_V = sqrt(X^T V X) " with " V in "diag"(R^p)$

    //    This gives more importance to some features than others.
  ]
  #only(5)[
    $w^(*) &= "argmin"_(w) ||X_("treat") - X_("control") w||_V^2\
      &s.t. space w_j >= 0, \
      &sum_(j=2)^(n_0 + 1) w_j = 1$
  ]
]


#slide[
  = Synthetic controls: Why choose positive weights summing to one?

  #show: later
  == This is called interpolation (vs extrapolation)

  #figure(image("img/event_studies/extrapolation.png", width: 70%))

  #show: later
  == Interpolation enforces regularization, thus limits overfitting

  Same kind of regularization than L1 norm in Lasso: forces some coefficient to be zero.

  // (both are #link("https://en.wikipedia.org/wiki/Convex_optimization", [_optimization with constraints on a simplex_])).
]

#slide[
  = Synthetic controls: Extrapolation failure with unconstrained weights

  #set align(horizon)
  #toolbox.side-by-side(columns: (2fr, 2fr))[
    $p=2 T_0$ covariates:

    $X_j= vec(Y_(j, 1), ...,Y_(j, T_0), Z_(j, 1), ..., Z_(j, T_0))^T in R^(2T_0)$

    Y cigarette sales, Z cigarette prices.

    #only(2)[
      Model: $underbrace(X_("treat"), p times 1)  tilde underbrace(X_("control"), p times n_0) underbrace(w, n_0)$
    ]
    #only(3)[#alert("-> Simple linear regression estimated by OLS")]

    #only((3, 4, 5))[
      Prediction: $hat(Y)_("synth") = vec(Y_(t, j))_(t=1..T#linebreak()j=2..n_0+1) w$
    ]
  ][
    #show: later
    #figure(image("img/pyfigures/scm_california_vs_synth_lr.svg", width: 100%))
  ]
  #only(5)[=== 😭 Overfitting]
]


#slide[
  = Synthetic controls: How to choose the predictor weights $V$?

  1. Don't choose: set $V = I_(p)$, i.e. $||X||_V = ||X||_2$.

  #show: later
  2. Rescale by the variance of the predictors: #linebreak() $V = "diag"("var"(Y_(j, 1))^(-1), .., "var"(Y_(j, T_0))^(-1), "var"(Z_(j, 1))^(-1), .., "var"(Z_(j, T_0))^(-1))$.

  #show: later
  3. Minimize the pre-treatment mean squared prediction error (MSPE) of the treated unit:

  $"MSPE"(V) &= sum_(t=1)^(T_0) [Y_(1, t) - sum_(j=2)^(n_0+1) w_j^*(V) Y_(j, t)]^2 \
    &= || vec(Y_(1, t))_(t=1..T_0) - vec(Y_(j, t))^T_(j=2..n_0+1#linebreak()t=1..T_0) hat(w) ||_(2)^(2)$

  This solution is solved by running two optimization problems:
  - #alert[Inner loop] solving $w^*(V)= "argmin"_(w) ||X_("treat") - X_("control") w||_V^2$

  - #alert[Outer loop] solving $V^*= "argmin"_(V) "MSPE"(V)$
]

#slide[
  = Synthetic controls: estimation without the outer optimization problem
  #toolbox.side-by-side(columns: (1.5fr, 2fr))[

    Same coviarates: $X_j= vec(Y_(j, 1), ...,Y_(j, T_0), Z_(j, 1), ..., Z_(j, T_0))^T$

    Y cigarette sales, Z cigarette prices.


    SCM minization with $V = I_(p)$, hence, $||X||_V = ||X||_2$.
    #v(1em)

    $w^(*) &= "argmin"_(w) ||X_("treat") - X_("control") w||_2^2\
      &s.t. space w_j >= 0, \
      &sum_(j=2)^(n_0 + 1) w_j = 1$

  ][
    #show: later
    #figure(image("img/pyfigures/scm_california_vs_synth_wo_v.svg", width: 100%))
  ]

]


#slide[
  = Synthetic controls: estimation with the outer optimization problem

  #figure(image("img/pyfigures/scm_california_vs_synth_pysyncon.svg", width: 70%))
]

#slide[
  = Synthetic controls: inference

  === Variability does not come from the variability of the outcomes

  Indeed, aggregates are often not very noisy (once deseasonalized)...

  #show: later
  === ... but from the variability of the chosen control units

  Treatment assignment introduces more noise than outcome variability.

  #show: later

  @abadie2010synthetic introduced the placebo test to assess the variability of the synthetic control.

  There is also a modern approach on inference for SCM based on Conformal prediction @chernozhukov2021exact (see end of the slides for intuition).
]


#slide[
  = Synthetic controls: inference with Placebo tests
  === Idea of placebo tests, also called Fisher's Exact tests

  - Permute the treated and control exhaustively.

  - For each unit, we pretend it is the treated while the others are the control: we call it a placebo

  - Compute the synthetic control for each placebo: it should be close to zero.

]


#slide[
  = Example
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

#slide[
  = Example

  #toolbox.side-by-side()[
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

#slide[
  == An event affecting the outcome for the treated unit and only part of the controls

  Setup @degli2020can:

  #toolbox.side-by-side(
    [
      - Population: US states
      - Intervention: Stand Your Ground law in Florida (october 2005)
      - Comparator: Other states without SYG laws
      - Outcome: homicide rate
    ],
    [#figure(image("img/event_studies/scm_failure_map.png", width: 70%))],
  )

  #show: later
  If it has an impact on the outcome after the treatment:
  For state in [KS, MD, AL, CT, FL], there is a step change in the outcome after the treatment: $bb(1)[t>T_0]$
]

#slide[
  = Synthetic controls failure: appropriate controls
  === Focus only on states affected by the confounding events

  #figure(image("img/event_studies/scm_failure_appropriate_control.png", width: 50%))

  Comparison states: KS, MD, AL, CT -> also affected by the counfounding event.

  🤩 We would conclude to no effect of the treatment.
]

#slide[
  = Synthetic controls failure: data-driven controls
  === Focus on all comparison states

  #figure(image("img/event_studies/scm_failure_data_driven_controls.png", width: 50%))

  SCM matches pre-treatment trends, without taking into account the confounding event.

  😥 We would falsely conclude to a positive treatment effect.
]

#slide[
  = Synthetic controls: Take-away

  #toolbox.side-by-side()[
    == Pros
    - More convincing for parallel trends assumption.
    - Handle multiple time periods.
    - Data driven.
    - Gives confidence intervals thanks to placebo test.
  ][
    #show: later
    == Cons
    - Many controls needed for good pre-treatment fits.
    - Prone to overfitting during the pre-treatment period.

    - Strong assumption: weights should balance the post-treatment unexposed outcomes i.e. conditional ignorability.
    - Still requires the no-anticipation assumption.
  ]
  #show: later
  See @arkhangelsky2021synthetic for discussions.
]

#new-section["Interrupted time-series: methods without a control group"]

#slide[
  = Interrupted Time Series: intuition
  == Setup

  - One #c_treated[treated unit], no #c_control[control unit].
  - Multiple time periods.
  - Sometimes, predictors are availables: there are called exogeneous covariates.

  #show: later
  == Intuition

  - Model the pre-treatment trend: $Y_t(1) "for" t<T_0$
  - Predict post-treatment trend as the control: $#c_control[$hat(Y_t)$(0)] "for" t>T_0 $

  - Obtain treatment effect by taking the difference between observed and predicted post-treatment observations: $#c_treated[$Y_(t)(1)$] - #c_control[$hat(Y_t)$(0)]$
]

#slide[
  = Interrupted Time Series: illustration from @schaffer2021interrupted

  #figure(image("img/event_studies/its_illustration.png", width: 65%))

  #set text(size: 18pt)
  $Y_t$: Dispensations of quetiapine, an anti-psychotic medicine.

  Treatment: Restriction of the conditions under which quetiapine could be subsidised.

]

#slide[

  = Modelization of a time-series

  == Tools

  - ARIMA models: AutoRegressive Integrated Moving Average


  == Motivation of ARIMA

  - Structure of autodependance between observation (auto-regression, moving average),
  - Linear trends,
  - Seasonality.

  === Good reference

  #link("https://otexts.com/fpp3/", "Forecasting (fpp3): Principles and Practice, chapter 8")
]

#slide[
  = ARIMA are State Space Models (SSM) #text("says the machine learning community",size: 18pt)

  == What is a (linear) state space model?

  #toolbox.side-by-side(columns: (2fr, 1fr))[

    - Two (sometimes multi-dimensional) components: the state $mu_t$ and the observation $y_t$.

    #show: later
    - State, ie. latent (unobserved) variable:
    #align(center)[
      $mu_t = overbrace(T_t, "Transition matrix") mu_(t-1) + overbrace(R_t, "Transition matrix") underbrace(eta_t, "gaussian white noise")$
    ]

    #show: later
    - Observation is a noisy version of the state:
    #align(center)[
      $y_t = overbrace(Z_t, "design matrix") mu_t + epsilon_t$
    ]
  ][
    #show: later
    #align(center)[
      #diagram(
        cell-size: 20mm,
        node-stroke: 0.6pt,
        node-shape: circle,
        spacing: 1em,
        let eta_t = (0, -1),
        let eta_t1 = (-1, -1),
        let mu_t = (0, 0),
        let mu_t1 = (-1, 0),
        let y_t = (0, 1),
        let y_t1 = (-1, 1),
        let epsilon_t1 = (-1, 2),
        let epsilon_t = (0, 2),
        node(eta_t, radius: 8mm, [$eta_t$]),
        node(eta_t1, radius: 8mm, [$eta_(t-1)$]),
        node(mu_t, radius: 8mm, [$mu_t$]),
        node(mu_t1, radius: 8mm, [$mu_(t-1)$]),
        node(y_t, radius: 8mm, [$y_(t)$]),
        node(y_t1, radius: 8mm, [$y_(t-1)$]),
        node(epsilon_t, radius: 8mm, [$epsilon_(t)$]),
        node(epsilon_t1, radius: 8mm, [$epsilon_(t-1)$]),
        edge(eta_t1, mu_t1, "->"),
        edge(eta_t1, mu_t, "->"),
        edge(eta_t, mu_t, "->"),
        edge(mu_t, y_t, "->"),
        edge(mu_t1, mu_t, "->"),
        edge(mu_t1, y_t1, "->"),
        edge(mu_t1, y_t, "->"),
        edge(epsilon_t, y_t, "->"),
        edge(epsilon_t1, y_t1, "->"),
      )
    ]
  ]

]

#slide[

  = Why showing the state space model formulation?

  - I better understand ARIMA formulated as state space models.

  - SSM are more general than ARIMA models.

  - ARIMA are (often) fitted with SSM optimization algorithms.

  === #alert[Good reference]
  @murphy2022probabilistic[book 2, chap 29]
  s
]


#slide[
  = State space models: AR(1) model example

  == AR(1)
  #set align(horizon)
  #toolbox.side-by-side(
    [
      #align(center)[
        == DAG
        #diagram(
          cell-size: 30mm,
          node-stroke: 0.6pt,
          node-shape: circle,
          spacing: 1em,
          let (y_1, y_2, y_3) = ((0, 0), (1, 0), (2, 0)),
          let (mu_1, mu_2, mu_3) = ((0, -1), (1, -1), (2, -1)),
          let (eta_1, eta_2, eta_3) = ((0, -2), (1, -2), (2, -2)),
          node(y_1, radius: 8mm, [$y_(t-1)$]),
          node(y_2, radius: 8mm, [$y_t$]),
          node(y_3, radius: 8mm, [$y_(t+1)$]),
          node(mu_1, radius: 8mm, [$mu_(t-1)$]),
          node(mu_2, radius: 8mm, [$mu_t$]),
          node(mu_3, radius: 8mm, [$mu_(t+1)$]),
          node(eta_1, radius: 8mm, [$eta_(t-1)$]),
          node(eta_2, radius: 8mm, [$eta_t$]),
          node(eta_3, radius: 8mm, [$eta_(t+1)$]),
          edge(mu_1, y_1, "->"),
          edge(mu_2, y_2, "->"),
          edge(mu_3, y_3, "->"),
          edge(mu_1, mu_2, "->"),
          edge(mu_2, mu_3, "->"),
          edge(eta_1, mu_1, "->"),
          edge(eta_2, mu_2, "->"),
          edge(eta_3, mu_3, "->"),
        )
      ]
    ],
    [
      == Formalization
      Latent: $mu_t = rho mu_(t-1) + eta_t$\
      Observation: $y_t =  mu_t $


      $"with" &eta_t ~ N(0, sigma^2)\
        &|rho|< 1$
    ],
  )
  #only(1)[
    Auto-regression time series model an outcome as a linear regression of its prior values.
  ]
]

#slide[
  = State space models: AR(2) model example

  == AR(2)
  #set align(horizon)
  #toolbox.side-by-side(
    [
      #align(center)[
        == DAG
        #diagram(
          cell-size: 30mm,
          node-stroke: 0.6pt,
          node-shape: circle,
          spacing: 1em,
          let (y_1, y_2, y_3) = ((0, 0), (1, 0), (2, 0)),
          let (mu_1, mu_2, mu_3, mu_4) = ((0, -1), (1, -1), (2, -1), (3, -1)),
          let (eta_1, eta_2, eta_3) = ((0, -2), (1, -2), (2, -2)),
          node(y_1, radius: 8mm, [$y_(t-1)$]),
          node(y_2, radius: 8mm, [$y_t$]),
          node(y_3, radius: 8mm, [$y_(t+1)$]),
          node(mu_1, radius: 8mm, [$mu_(t-1)$]),
          node(mu_2, radius: 8mm, [$mu_t$]),
          node(mu_3, radius: 8mm, [$mu_(t+1)$]),
          node(eta_1, radius: 8mm, [$eta_(t-1)$]),
          node(eta_2, radius: 8mm, [$eta_t$]),
          node(eta_3, radius: 8mm, [$eta_(t+1)$]),
          edge(mu_1, y_1, "->"),
          edge(mu_2, y_2, "->"),
          edge(mu_3, y_3, "->"),
          edge(mu_1, mu_2, "->"),
          edge(mu_1, mu_3, "->", bend: 30deg),
          edge(mu_2, mu_3, "->"),
          edge(mu_2, mu_4, "->", bend: 30deg),
          edge(eta_1, mu_1, "->"),
          edge(eta_2, mu_2, "->"),
          edge(eta_3, mu_3, "->"),
        )
      ]
    ],
    [
      == Formalization
      Latent: $mu_t = mat(rho_1, rho_2; 1, 0) mu_(t-1) + vec(1, 0) eta_t$\
      Observation: $y_t =  [1, 0] mu_t $\

      $"with" &eta_t ~ N(0, sigma^2)\
        &|rho_1|< 1, |rho_2|< 1$

      Observation unrolled:\
      $y_t = rho_1 y_(t-1) + rho_2 y_(t-2) + eta_t$
    ],
  )
]

#slide[
  = State space models: MA(1) i.e. ARIMA(0,0,1) model example

  #toolbox.side-by-side(
    [
      #align(center)[
        == DAG
        #diagram(
          cell-size: 30mm,
          node-stroke: 0.6pt,
          node-shape: circle,
          spacing: 1em,
          let (y_1, y_2, y_3) = ((0, 0), (1, 0), (2, 0)),
          let (mu_1, mu_2, mu_3) = ((0, -1), (1, -1), (2, -1)),
          let (eta_1, eta_2, eta_3) = ((0, -2), (1, -2), (2, -2)),
          node(y_1, radius: 8mm, [$y_(t-1)$]),
          node(y_2, radius: 8mm, [$y_t$]),
          node(y_3, radius: 8mm, [$y_(t+1)$]),
          node(mu_1, radius: 8mm, [$mu_(t-1)$]),
          node(mu_2, radius: 8mm, [$mu_t$]),
          node(mu_3, radius: 8mm, [$mu_(t+1)$]),
          node(eta_1, radius: 8mm, [$eta_(t-1)$]),
          node(eta_2, radius: 8mm, [$eta_t$]),
          node(eta_3, radius: 8mm, [$eta_(t+1)$]),
          edge(mu_1, y_1, "->"),
          edge(mu_2, y_2, "->"),
          edge(mu_3, y_3, "->"),
          edge(eta_1, mu_1, "->"),
          edge(eta_1, mu_2, "->"),
          edge(eta_2, mu_2, "->"),
          edge(eta_2, mu_3, "->"),
          edge(eta_3, mu_3, "->"),
        )
      ]
    ],
    [
      == Formalization

      Latent: $mu_t= [1, theta] vec(eta_t, eta_(t-1))$\
      Observation: $y_t = mu_t$

      $"with" &eta_t ~ N(0, sigma^2)$
    ],
  )
  #show: later
  The MA time series models the residual of the regression of $y_t$ on its previous values as a linear combination of the previous residuals : i.e. vanishing shocks.
]

#slide[

  = State space models: ARMA(p, q) i.e. ARIMA(p,0,q) model example

  == Formalization (Hamilton form)
  Let $r = max(p, q+1)$

  Observation: $y_t = (1, theta_1, theta_2, ..., theta_(r-1)) \u{0020} mu_t$

  Latent: $mu_t = mat(
    1, rho_1, rho_2, ..., rho_(r-1);
    1, 0, 0, ..., 0;
    0,1, 0, ..., 0;
    dots.v, dots.down, dots.v, dots.v, dots.v;
    0, ..., 0, 1, 0;
    ) mu_(t-1) + vec(epsilon_(t), 0, ..., 0)$ with $epsilon_(t) ~ N(0, sigma^2)$


  == Unfolding the state space equations
  $y_t = sum_(i=1)^(p) rho_i y_(t-i) + sum_(j=1)^(q) theta_j epsilon_(t-j)$
]


#slide[
  = State space models: Adding a seasonnality and a covariate component

  #toolbox.side-by-side(
    [
      #align(center)[
        == DAG
        #diagram(
          cell-size: 12mm,
          node-stroke: 0.6pt,
          node-shape: circle,
          spacing: 1em,
          let (s_11, s_12, s_13) = ((0, 0), (0, -1), (0, -2)),
          let (s_1, s_2, s_3) = ((1, 0), (1, -1), (1, -2)),
          let (mu_1, mu_2) = ((0, -3), (1, -3)),
          let (eta_1, eta_2) = ((0, -4), (1, -4)),
          let y = (1, 1),
          let x = (2, 1),
          node(s_11, radius: 7mm, [$s_(t-1)$]),
          node(s_12, radius: 7mm, [$s_(t-2)$]),
          node(s_13, radius: 7mm, [$s_(t-3)$]),
          node(s_1, radius: 7mm, [$s_(t)$]),
          node(s_2, radius: 7mm, [$s_(t-1)$]),
          node(s_3, radius: 7mm, [$s_(t-2)$]),
          node(mu_1, radius: 7mm, [$mu_(t-1)$]),
          node(mu_2, radius: 7mm, [$mu_t$]),
          node(eta_1, radius: 7mm, [$eta_(t-1)$]),
          node(eta_2, radius: 7mm, [$eta_t$]),
          node(y, radius: 7mm, [$y_t$]),
          node(x, radius: 7mm, [$x_t$]),
          edge(s_11, s_1, "->"),
          edge(s_11, s_2, "->"),
          edge(s_12, s_1, "->"),
          edge(s_12, s_3, "->"),
          edge(s_13, s_1, "->"),
          edge(s_1, y, "->"),
          edge(mu_1, mu_2, "->"),
          edge(eta_1, mu_1, "->"),
          edge(eta_1, mu_2, "->"),
          edge(eta_2, mu_2, "->"),
          edge(mu_2, y, "->", bend: 40deg),
          edge(x, y, "->"),
        )

      ]
    ],
    [
      == Formalization

      Observation with covariates and seasonality:

      $y_t = beta x_t + s_t + underbrace(rho mu_(t-1), "AR(1)") + underbrace(theta eta_(t-1) + eta_t, "MA(1)")$

      #v(1em)
      Where seasonality:

      $s_t &= - sum^(S-1)_(k=1) s_(t-k) + epsilon_(s,t)\
        &"with" epsilon_s,t ~ N(0, sigma^2_s)$
    ],
  )
]

#slide[
  = State space models: General formulation


  #toolbox.side-by-side(
    columns: (1fr, 2fr),
    [
      Latent:\ $mu_t = T_t mu_(t-1) + R_t eta_t$\

      Observation:\ $y_t = Z_t mu_t + beta^T x_t + epsilon_t$

      With $eta_t$ and $epsilon_t$ mean zero gaussian noise, sometimes with a specific covariance structure.
    ],
    [
      #figure(image("img/event_studies/complex_ssm.png", width: 90%))
    ],
  )
  Complex SSM DAG from the Causal Impact paper @brodersen2015inferring.
]

#slide[
  = State space models: a brief word on fitting (i.e. learning the parameters)
  === When the error terms are gaussians

  These modeles are called linear Gaussian state space model (LG-SSM) or linear dynamical system (LDS).

  === The likelihood is jointly gaussian

  #alert[Closed form formula] for the likelihood of the data under the model.

  #show: later

  == Expectation-Minimization: a widespread algorithm for fitting

  - Expectaction: Compute the joint likelihood of the data and the parameters (observed outcome, unknown state) given the parameters.

  - Maximization: find parameters maximizing the likelihood: analytically since gaussian.

  - Iter until convergence to a (local) maximum of likelihood.
]

#slide[
  = Modern state space models

  === Long Short Term Memory (LSTM) networks
  A type of Recurrent Neural Network (RNN) that can learn long-term dependencies @graves2012long.

  It was state of the art for language tasks before transformers.

  It is notably hard to train due to vanishing gradient through the time dimension.

  === Mamba
  A recent proposition to mitigate one of the main limitations of the transformer architecture: high complexity relative to the length of the sequence @gu2023mamba.

  Good blog-style introduction in @Ayonrinde2024mamba.
]


#slide[
  = Example of ITS with ARIMA: the French antibiotics campaign of 2002-2007
  == Context

  In 2001, compared to the European Union countries, France was a country where:
  - the population consumed the most antibiotics in town
  - the resistance of Streptococcus pneumoniae to β-lactams was the highest (53%)
  - a significant number of antibiotic prescriptions would be unnecessary (viral infections)

  == Campaign (october 2002, then every year october to march)

  France implemented a national plan to “preserve the effectiveness of antibiotics and improve their use” with the main action undertaken by the National Health Insurance.


  == Question

  What has been the effect of the campaign on the consumption of antibiotics? @sabuncu2009significant
]

#slide[
  = Example of ITS with ARIMA: the French antibiotics campaign of 2002-2007
  == Weekly reimbursed prescription of antibiotics in town

  #figure(image("img/event_studies/sabuncu2009_fig1.png", width: 70%))

  Interventions during the months of october to march: $"month"(t) in M_0$.
]


#slide[
  = Example of ITS with ARIMA: the French antibiotics campaign of 2002-2007
  == Estimation

  - Fit an ARIMA model on the pre-treatment trend

  - Introduce an additive term for the intervention: #linebreak()
  $Y_t = c + sum_i hat(tau_i) bb(1)["month"(t) in M_0 and "year"(t)==i] + underbrace([a(B)^(-1) - b (B) epsilon_s], "ARIMA term fitted on pre-treatment")$

  - Assess if the additive term and other parameters are significantly different pre-treatment and post-treatment.
]

#slide[
  = Example of ITS with ARIMA: the French antibiotics campaign of 2002-2007

  #figure(image("img/event_studies/sabuncu2009_fig4.png", width: 64%))
  #set text(size: 18pt)
  - #text(red)[Red curve: arima fitted with intervention]
  - #text(red)[Red Horizontal line: intervention effect fitted during intervention]

  - Black curve: arima fitted without intervention
  - Black horizontal line: intervention effect fitted pre-intervention

]

// #slide(title: "Example of ITS with more general SSM: Causal impact")[
//   TODO
// ]


#slide[
  = A word on model families for ITS
  We saw ARIMA models and the more general class of state space models.

  However, we could any model that we want to fit the pre-treatment trend !

  #show: later
  #link("https://facebook.github.io/prophet/", [Facebook prophet model]) @taylor2018forecasting uses Generalized Additive Models (GAM).

  #show: later
  Any sklearn estimator could do the trick: Linear regression, Random Forest, Gradient Boosting...

  #show: later
  ⚠️ You should pay attention to appropriate train/test split when cross-validating a time-series model not to use the future to predict the past.

  Relevant remark for all time series models (even ARIMA or state space models).
]

#slide[

  = Cross-validation for time-series models

  ```python
  from sklearn.model_selection import TimeSeriesSplit
  ```
  #figure(image("img/event_studies/time_series_cv.png", width: 70%))

  This avoids to use the future to predict the past.
]


#slide[
  = Main threat to validity for an ITS: historical bias
  @degli2020can[Fig. 1]

  #toolbox.side-by-side()[
    ⚠ If there is a co-intervention, it will impact the outcome trend and bias the treatment effect estimation.

    #uncover(2)[
      💡 Adding a control series of predictors can help to mitigate this bias.
    ]
  ][
    #only(1)[
      #figure(image("img/event_studies/its_cofounding_event.png", width: 100%))
    ]
    #only(2)[
      #figure(image("img/event_studies/its_cofounding_event_controlled.png", width: 100%))
    ]
    //Illustration from
  ]
]


#slide[
  = Take-away on ITS

  == Pros

  - Suitable when no control unit is available. The pre-treatment trend is the control.

  - Handles multiple time periods.

  - A lot of software available (eg. ARIMA models).

  - Simple: few parameters to tune.

  == Cons

  - Prone to bias by other events happening around the treatment time and impacting the outcome trend.

  - Prone to overfitting of the pre-treatment trend.
]


#slide[
  = An attempt to map event study methods

  #set text(size: 15pt)

  #table(
    columns: 5,
    align: (left, left, center, center, center),
    table.header("Methods", "Characteristics", "Hypotheses", "Community", "Introduction"),
    "DID/TWFE",
    [Treated/control units, few time periods, no predictors],
    [Parallel trends, no anticipation, prone to overfitting],
    [Economics],
    [#link(
        "https://matheusfacure.github.io/python-causality-handbook/13-Difference-in-Differences.html",
        "Causal Inference for the Brave and True, chapter 13",
      )],
    //[@arkhangelsky2024causal],

    "ARIMA, ITS",
    [No controls, no/few predictors, seasonality],
    [Stationnarity , no anticipation, prone to overfitting],
    [Epidemiology, Economics],
    [#link("https://otexts.com/fpp3/arima.html", "Forecasting: Principles and Practice")],
    // [@schaffer2021interrupted],

    "State space models",
    [Multiple time periods, control units or predictors, generalization of ARIMA],
    [Contional ignorability on predictors, goodness of fit pre-treatment],
    [Machine learning, bayesian methods],
    [#link(
        "repository.cinec.edu/bitstream/cinec20/1109/1/2016_Book_IntroductionToTimeSeriesAndFor.pdf#page=259",
        [Introduction
          to Time Series
          and Forecasting, chapter 9],
      )],
    // [@murphy2022probabilistic[chapter 18]],

    "Synthetic control",
    [Treated/control units, multiple time periods],
    [Conditional parallel trend on controls, goodness of fit pre-treatment],
    [Economics],
    [#link(
        "https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html",
        "Causal Inference for the Brave and True, chapter 15",
      )],
    // [@abadie2021using],
  )

]

#slide[
  = A summary on R packages for event studies

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


#slide[
  = A summary on Python packages for event studies

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


#slide[
  = Final word -- What methods to chose: some guides

  #set text(size: 18pt)
  == DID-family methods

  - Control units available (at least one)
  - Few time periods
  - Parallel trend is credible (if necessary by adjusting the model on predictors).

  == Synthetic Control Methods

  - Mutiple and different controls as well as multiple time periods
  - Pre-treatment outcomes of the control units predict well the treated unit outcome.
  - No-spill over from the treatment to the control units.

  == ITS: SARIMA or state space models

  - No evident control units
  - Pre-treatment outcome of the treated unit seems a good control
  - Control predictors not impacted by the treatment availables
  - No co-intervention that could impact the treated outcome.
]


#new-section["Python hands-on"]

#slide[
  = To your notebooks 🧑‍💻!
  - url: https://github.com/strayMat/causal-ml-course/tree/main/notebooks
]

#new-section["Supplementary materials"]


#slide[
  = Synthetic controls: conformal prediction inference

  Introduced by @chernozhukov2021exact

  - Recast the problem as #alert[counterfactual inference], i.e. predict: $Y_(i t)(0) "for" t>T_0 $

  - Test hypothesis: $H_0$ eg. $H_0 = (0, 0, .., 0)$ ie no effect for $t > T_0$

  - This imply the generation of a hypothesis counterfactual trajectory $Y_t (0)$

  == Question

  Are the post-treatment residuals of a model fitted on the hypothesis counterfactual trajectory an outlier of the distribution of the residuals pre-treatment?

  == Why does this works?

  Syntehtic controls estimation are invariant under the time series dimension so we can resample under this dimension to introduce data variability.

]

#slide[
  = Conformal inference: hypothesis generation
  - Test a hypothesis : $H_0$ eg. $H_0 = (0, 0, .., 0)$ ie no effect for $t > T_0$

  - Gerenate a counterfactual trajectory $Y_t (0)$ under this null
]

#slide[
  = Conformal inference: Fit a model and compute residuals
  - Fit a counterfactual model on the #alert[full generated trajectory]: $hat(Y_t)$

  - Compute the residuals: $hat(u)_t = Y_t (0) - hat(Y)_t$
]

#slide[
  = Conformal inference: test statistic and resampling
  Summarize the residuals in a statistic: $S(hat(u)) = big("(") 1 / sqrt(T-T_0 + 1) sum_(t=T_0 + 1)^(T) |hat(u)_t|^q big(")")^(1/q)$

]


#slide[
  = Conformal inference: resampling
  == Resample this statistic by block permutation $pi$ of the time periods

  Same as permutting the data since SCM are invariant under the time series dimension.

  #figure(image("img/event_studies/block-perm.png", width: 60%))

  #align(center)[#text(size: 12pt)[Image from: Causal Inference for the Brave and True]]
]

#slide[
  = Conformal inference: P-value
  - Assess if the post-treatment statistics is an outlier of this distribution.

  - P-value: $hat(F) (x) = 1/(|Pi|) sum_(pi in Pi) bb(1) [S(hat(u)_(pi_(0))) <= S(hat(u)_(pi))]$ where $pi_(0)$ is the original data.

]

// #slide[
//   = Informal inference: confidence intervals
//   TODO
// ]


// #slide[
//   = Conditional difference-in-differences
//   TODO
// ]




#set align(left)
#set text(size: 20pt, style: "italic")

#slide[
  = Bibliography
  #bibliography
]
