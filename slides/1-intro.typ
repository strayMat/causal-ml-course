// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.3.1": *

#import themes.metropolis: *

#show: metropolis-theme.with(footer: [ENSAE, Introduction course])

// Make the paper dimensions fit for a presentation and the text larger
#set page(paper: "presentation-16-9")
#set text(size: 22pt)
#set figure(numbering: none)
#show math.equation: set text(font: "Fira Math")
#set strong(delta: 100)
#set par(justify: true)
#show figure.caption: set text(size: 18pt)

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
  subtitle: "Causal perspective",
  date: "January 10, 2025",
  //extra: "Extra"
)

#slide(title: "Table of contents")[
  #metropolis-outline
]

#new-section-slide("Introduction")

#slide(
  title: "Causal inference: subfield of statistics dealing with \"why questions\"",
)[

  #set align(top + center)
  #image("img/intro/confounder.svg", width: 15%)

  #set align(top + left)

  At the center of epidemiology, econometrics, social sciences, #only(2)[machine learning]...

  #only(
    2,
  )[
    Now, bridging with machine learning @kaddour2022causal : Fairness, reinforcement
    learning, causal discovery, causal inference for LLM, causal representations... //#footnote[Eg. #link("https://nips.cc/virtual/2024/events/workshop")[neurips 2024 workshops]]
  ]
  #uncover(
    3,
  )[
    This course: #alert[Basis of causal inference using ML appraoches (semi-parametric), inspiration
      from epidemiology and application for applied econometrics.]
  ]
]

#slide(
  title: "What is a \"why question\"?",
)[

  - Economics: How does supply and demand (causally) depend on price?

  - Policy: Are job training programmes actually effective?

  - Epidemiology: How does this threatment affect the patient's health?

  - Public health : Is this prevention campaign effective?

  - Psychology: What is the effect of family structure on children's outcome?

  - Sociology: What is the effect of social media on political opinions?
]

#slide(title: "This is different from a predictive question")[

  - What will be the weather tomorrow?
  - What will be the outcome of the next election?
  - How many people will get infected by flue next season?
  - What is the cardio-vacular risk of this patient?
  - How much will the price of a stock be tomorrow?
]

#slide(
  title: "Why is prediction different from causation? (1/2)",
)[
  - Prediction (most part of ML): What usually happens in a given situation?
  #uncover(
    2,
  )[
    #alert([Assumption]) Train and test data are drawn from the same distribution.

    #side-by-side(gutter: 3mm, columns: (1fr, 1fr))[
      #figure(image("img/intro/dag_x_to_y.svg", width: 40%))
    ][
      Prediction models $(X, Y)$
    ]
  ]
]

#slide(
  title: "Why is prediction different from causation? (2/2)",
)[
  - Causal inference (most part of economists) : What would happen if we changed the
    system ie. under intervention?
  #uncover(
    2,
  )[
    #alert([Assumption]): No unmeasured variables influencing both treatment and
    outcome $arrow$ confounders.
    #side-by-side(
      gutter: 3mm,
      columns: (1fr, 1fr),
    )[
      #figure(image("img/intro/confounder.svg", width: 40%))
    ][
      Causal inference models $(X, A, Y(A=1), Y(A=0))$, the covariate shift between
      treated and control units.
    ]
  ]
]

#slide(
  title: "Machine learning is pattern matching (ie. curve fitting)",
)[
  Find an estimator $f : x arrow y$ that approximates the true value of y so that $f(x) approx y$
  #figure(
    image("img/intro/ml_curve_fitting.png", width: 55%),
    caption: "Boosted trees : iterative ensemble of decision trees",
  )
]

#slide(
  title: "Machine learning is pattern matching that generalizes to new data",
)[
  Select models based on their ability to generalize to new data : (train, test)
  splits and cross validation @stone1974cross.

  #figure(
    image("img/intro/cross_validation.png", width: 50%),
    caption: ["Cross validation" @varoquaux2017assessing],
  )

]

//TODO: insert images of pattern matching and covariate shifts to illustrate.

#new-section-slide("How to ask a sound causal question: The PICO framework")
//lalonde example from the book or wage gap example (I am less fan of no intervention examples).

#slide(
  title: "Identify the target trial",
)[
  ==
  What would be the ideal *randomized experiment* to answer the question?
  #cite(<hernan2016using>)

]

#slide(title: "PICO framework")[
  - Population : Who are we interested in?
  - Intervention : What treatment/intervention do we study?
  - Comparison : What are we comparing it to?
  - Outcome : What are we interested in?
  //- Define the causal measure //a bit too much for intro
]

#slide(
  title: "PICO framework, an illustration",
)[
  #set text(size: 16pt)
  #table(
    columns: (auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header([*Component*], [*Description*], [*Notation*], [*Example*]),
    "Population",
    "What is the target population of interest?",
    $ X ∼ P(X) $,
    "Patients with sepsis in the ICU",
    "Intervention",
    "What is the treatment?",
    $ A ∼ P(A = 1) = p_A $,
    "Crystalloids and albumin combination",
    "Control",
    "What is the clinically relevant comparator?",
    $ 1 - A ∼ 1 - p_A $,
    "Crystalloids only",
    "Outcome",
    "What are the outcomes?",
    $ Y(1), Y(0) ∼ P(Y(1), Y(0)) $,
    "28-day mortality",
    "Time",
    "Is the start of follow-up aligned with intervention assignment?",
    "N/A",
    "Intervention administered within the first 24 hours of admission",
  )

]

#new-section-slide("Causal graphs")

#slide(title: "Directed acyclic graphs (DAG): reason about causality")[
  What are the important depedencies between variables?
]

#new-section-slide(
  "Four steps of causal inference : identification, estimand, causal and statistical inference, vibration analysis",
)

#slide(title: "Causal estimand")[
  What can we learn from the data?
]

#slide(
  title: "Identification",
)[

  What can we learn from the data?

  Knowledge based

  Cannot be validated with data //#uncover(2)[Still, there is some work on causal discovery, mostly based on conditional independence tests @glymour2019review.]
]

#new-section-slide("Potential outcomes")

#new-section-slide("Causal inference")

#slide(
  title: "PICO framework and the potential outcomes",
)[
  #set text(size: 16pt)
  #table(
    columns: (auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header([*Component*], [*Description*], [*Notation*], [*Example*]),
    "Population",
    "What is the target population of interest?",
    $ X ∼ P(X) $,
    "Patients with sepsis in the ICU",
    "Intervention",
    "What is the treatment?",
    $ A ∼ P(A = 1) = p_A $,
    "Crystalloids and albumin combination",
    "Control",
    "What is the clinically relevant comparator?",
    $ 1 - A ∼ 1 - p_A $,
    "Crystalloids only",
    "Outcome",
    "What are the outcomes?",
    $ Y(1), Y(0) ∼ P(Y(1), Y(0)) $,
    "28-day mortality",
    "Time",
    "Is the start of follow-up aligned with intervention assignment?",
    "N/A",
    "Intervention administered within the first 24 hours of admission",
  )
]

#new-section-slide("Statistical estimand")

#new-section-slide("Statistical inference ie. estimation")

#new-section-slide("Related concepts")

#set align(horizon + center)

- Structural equations:

#slide[
  == Resources

  - https://web.stanford.edu/~swager/stats361.pdf
  - https://www.mixtapesessions.io/
  - https://alejandroschuler.github.io/mci/
]

#let bibliography = bibliography("biblio.bib", style: "apa")

#set align(left)
#set text(size: 20pt, style: "italic")
#slide[
  #bibliography
]