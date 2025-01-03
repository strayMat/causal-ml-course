// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.3.1": *
#import "@preview/embiggen:0.0.1": * // LaTeX-like delimiter sizing for Typst
#import "@preview/showybox:2.0.1": showybox

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
    align(center, body),
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

  == Estimation of the effect of a treatment when data is


  Aggregated: eg. country-level data such as employment rate, GDP.

  #pause

  Longitudinal: eg. multiple time periods or repeated cross-sections.

  #pause

  With multiple units: eg. multiple countries, firms, regions.

  #pause

  Staggered adoption of the treatment: eg. different countries adopt a policy at different times.

  #pause

  This setup is known as: #alert[panel data, event studies, longitudinal data, time-series data.]
]

#slide(title: "Examples of event studyies for policy question")[

]

#slide(title: "Setup: event studies are quasi-experiment")[
  - #link("https://en.wikipedia.org/wiki/Quasi-experiment", "Quasi-experiment"): a situation where the treatment is not randomly assigned by the researcher but by nature or society.

  - Should introduces some randomness in the treatment assignment: enforcing treatment exogeneity, ie. ignorability (ie. unconfoundedness).
  //#TODO causal diagram with times

  == Today: Three quasi-experimental designs for event studies

  - The simple method of difference-in-differences with a strong assumption called paralled trend

  - Synthetic control method: a balancing method (think to propensity score matching)

  - Conditional DID: a doubly robust method combining outcomes and propensity score models
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
  == This is called interpolation (!= extrapolation)

  #figure(image("img/event_studies/extrapolation.png", width: 70%))

  #only(2)[
    == Interpolation enforces regularization, thus limits overfitting

    Same kind of regularization than L1 norm in Lasso: forces some coefficient to be zero (both are #link("https://en.wikipedia.org/wiki/Convex_optimization", [_optimization with constraints_])).
  ]
]

#slide(title: "Synthetic controls: Extrapolation failure with unconstrained weight")[
  #set align(horizon)
  #side-by-side(columns: (2fr, 2fr))[
    $p=2 T_0$ covariates:

    $X_j= vec(Y_(j, 1), ..,Y_(j, T_0), Z_(j, 1), .., Z_(j, T_0))^T in R^(2T_0)$

    Y cigarette sales, Z cigarette prices.

    #pause
    Model: $underbrace(X_("treat"), p times 1)  tilde underbrace(X_("control"), p times n_0) underbrace(w, n_0)$

    #pause
    Prediction: $hat(Y)_("synth") = vec(Y_(j, t))_(T times n_0) w in RR^T$
  ][
    #pause
    #figure(image("img/pyfigures/scm_california_vs_synth_lr.svg", width: 100%))
  ]
  #only(5)[=== 😭 Overfitting]
]



#new-section-slide("Conditional difference-in-differences")


#new-section-slide("Time-series modelisation: methods without a control group")

#slide(title: "Interrupted Time Series")[
  == Idea

  - Compare the evolution of the outcome before and after the treatment
  - The treatment effect is the difference between the two trends

  == Example
  -
]


#slide(title: "State space models")[

]


#slide(title: "Take-away")[

]

#slide(title: "Good references for event studies")[

  - The causal mixtape: #link("https://mixtape.scunning.com/09-difference_in_differences")

  - Causal inference for the brave and true: #link("https://matheusfacure.github.io/python-causality-handbook/13-Difference-in-Differences.html")
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
