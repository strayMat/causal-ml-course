// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.3.1": *
#import "@preview/embiggen:0.0.1": *
#import "@preview/showybox:2.0.1": showybox
#import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge // for dags

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
      top: 0em,
      right: 1em,
      bottom: 1em,
    ) // super important to have a proper padding (not 1/3 of the slide blank...)
    set text(fill: m-dark-teal)
    body
  }

  logic.polylux-slide(content)
}


#let imp_block(title: "", body) = {
  showybox(
    frame: (
      border-color: red.darken(50%),
      title-color: red.lighten(60%),
      body-color: red.lighten(80%),
    ),
    title-style: (
      color: black,
      weight: "regular",
      align: center,
    ),
    title: title,
    body,
  )
}

#let ex_block(title: "Example", body) = {
  showybox(
    title-style: (
      color: black,
      weight: "regular",
      boxed-style: (
        anchor: (
          x: center,
          y: horizon,
        ),
        radius: 10pt,
      ),
    ),
    frame: (
      border-color: green.darken(50%),
      title-color: green.lighten(60%),
      body-color: green.lighten(80%),
    ),
    title: title,
    body,
  )
}



#title-slide(
  author: [Matthieu Doutreligne],
  title: "Machine Learning for econometrics",
  subtitle: [Reminders of potential outcomes and Directed Acyclic Graphs],
  date: "January 10, 2025",
  extra: "Thanks to Judith Abecassis for the slides on DAGs",
)

#slide(title: "Table of contents")[
  #metropolis-outline
]

#new-section-slide("Introduction")

#slide(title: "Causal inference: subfield of statistics dealing with \"why questions\"")[

  #grid(
    columns: (auto, auto),
    gutter: 3pt,
    image("img/po_reminder_and_dags/what_if.jpg", width: 40%),
    image("img/po_reminder_and_dags/causal_ml_book.png", width: 47%),
  )
  At the center of epidemiology @hernan2020causal, econometrics #cite(<chernozhukov2024applied>),
  social sciences, #only(2)[machine learning...]

  #only(2)[
    Now, bridging with machine learning @kaddour2022causal : Fairness, reinforcement
    learning, causal discovery, causal inference for LLM, causal representations... //#footnote[Eg. #link("https://nips.cc/virtual/2024/events/workshop")[neurips 2024 workshops]]
  ]
  #only(3)[
    #alert[This session:]
    - Reminder on causal inference
    - Importance of Directed Acyclic Graphs and causal status of covariates
  ]
]

#slide(title: "What is a \"why question\"?")[

  - Economics: How does supply and demand (causally) depend on price?

  #pause
  - Policy: Are job training programmes actually effective?

  #pause
  - Epidemiology: How does this threatment affect the patient's health?

  #pause
  - Public health : Is this prevention campaign effective?

  #pause
  - Psychology: What is the effect of family structure on children's outcome?

  #pause
  - Sociology: What is the effect of social media on political opinions?
]

#slide(title: "This is different from predictive questions")[

  == Prediction (ML): What usually happens in a given situation?

  #side-by-side(gutter: 3mm, columns: (1fr, 1fr))[
    #figure(image("img/po_reminder_and_dags/dag_x_to_y.svg", width: 40%))
  ][
    Prediction models $(X, Y)$
  ]

  #pause
  - What will be the weather tomorrow?

  #pause
  - What will be the outcome of the next election?

  #pause
  - How many people will get infected by flue next season?

  #pause
  - What is the cardio-vacular risk of this patient?

  #pause
  - How much will the price of a stock be tomorrow?

  #pause
  #align(center)[#alert([Assumption]) Train and test data are drawn from the same distribution.]

]

#slide(title: "Machine learning is pattern matching")[
  Find an estimator $f : x arrow y$ that approximates the true value of y so that $f(x) approx y$
  #figure(
    image("img/po_reminder_and_dags/ml_curve_fitting.png", width: 55%),
    caption: "Boosted trees : iterative ensemble of decision trees",
  )
]

#slide(title: "Machine learning is pattern matching that generalizes to new data")[
  Select models based on their ability to generalize to new data : (train, test)
  splits and cross validation @stone1974cross.

  #figure(
    image("img/po_reminder_and_dags/cross_validation.png", width: 50%),
    caption: ["Cross validation" @varoquaux2017assessing],
  )
]


#slide(title: "Machine learning is great for prediction on complex data")[

  #side-by-side(
    [
      #uncover((1, 2, 3))[
        - Images: Image classification with deep convolutional neural networks @krizhevsky2012imagenet
      ]

      #uncover((2, 3))[
        - Speech-to-text: Towards end-to-end speech recognition with recurrent neural networks @graves2014towards
      ]

      #uncover(3)[
        - Text: Attention is all you need @vaswani2017attention
      ]
    ],
    [
      #set align(center)
      #only(1)[
        #figure(
          image("img/po_reminder_and_dags/image_classification.png", width: 100%),
          caption: "ImageNet 1K: 1.5 million images, 1000 classes",
        )
      ]
      #only(3)[
        #figure(
          image("img/po_reminder_and_dags/ner_edsnlp.png", width: 100%),
          caption: "Named entity recognition",
        )
      ]
    ],
  )
]

#slide(title: "Machine learning might be less successful for what if questions")[
  == Machine learning is not driven by causal mechanisms

  - For example people that go to the hospital die more than people who do not #footnote[Example from #link("sklearn mooc")[https://inria.github.io/scikit-learn-mooc/concluding_remarks.html?highlight=causality]]:

  - Naive data analysis might conclude that hospitals are bad for health.

  - The fallacy is that we are comparing different populations: people who go to the hospital typically have a worse baseline health than people who do not.

  This is a *confounding factor*: a variable that influences both the treatment and the outcome.
]

#slide(title: "Why is prediction different from causation? (2/2)")[
  === Causal inference (most part of economists) : What would happen if we changed the system ie. under an intervention?
  #uncover(2)[
    #alert([Assumption]): No unmeasured variables influencing both treatment and
    outcome $arrow$ confounders.
    #side-by-side(gutter: 3mm, columns: (1fr, 2fr))[
      #diagram(
        cell-size: 10mm,
        node-stroke: 0.6pt,
        spacing: 1.5em,
        let (X, A, Y) = ((0, 0), (-1, 1), (1, 1)),
        node(A, "A"),
        node(Y, "Y"),
        node(X, "X"),
        edge(X, Y, "->"),
        edge(X, A, "->"),
        edge(A, Y, "->"),
      )

    ][
      Causal inference models #linebreak()
      $(X, A, Y(A=1), Y(A=0))$ #linebreak()
      the covariate shift between treated and control units.
    ]
  ]
]


#slide(title: [Illustration of the fundamental problem of causal inference])[
  == Example from epidemiology

  - Population: patients experiencing a stroke

  #pause
  - #c_treated[Intervention $A = 1$: patients had access to a MRI scan #text(weight: "extrabold")[in less than 3 hours] after the first symptoms]
  - #c_control[Comparator $A = 0$: patients had access to a MRI scan #text(weight: "extrabold")[in more than 3 hours] after the first symptoms]

  #pause
  - $Y = PP[text("Mortality")]$: the mortality at 7 days

  #pause
  - $X = PP[text("Charlson score")]$: a comorbidity index summarizing the overall health state of the patient. Higher is bad for the patient.

]


#slide(title: [Illustration: observational data])[
  == Draw a population sample without treatment status
  #figure(
    image("img/po_reminder_and_dags/sample_wo_oracles_gray.png", width: 83%),
    //caption: "DAG for a RCT: the treatment is independent of the confounders",
  )
]

#slide(title: [Illustration: observational data])[
  == Draw a population sample with treatment status
  #figure(
    image("img/po_reminder_and_dags/sample.svg", width: 75%),
    //caption: "DAG for a RCT: the treatment is independent of the confounders",
  )
]



#slide(title: [Illustration: observational data, a naive solution])[
  == Compute the difference in mean (DM): $tau_(text("DM"))=$ $$#c_treated[$EE[Y(1)]$] - #c_control($EE[Y(0)]$)

  #uncover(2)[
    #figure(
      image("img/po_reminder_and_dags/sample_mean_difference.png", width: 75%),
      //caption: "DAG for a RCT: the treatment is independent of the confounders",
    )
  ]
]

#slide(title: [RCT case: No problem of confounding])[

  #side-by-side(gutter: 10mm, columns: (1fr, 1fr))[
    #set align(center)
    #uncover((1, 2))[
      = Observational data

      #diagram(
        cell-size: 10mm,
        node-stroke: 0.6pt,
        spacing: 1.5em,
        let (X, A, Y) = ((0, 0), (-1, 1), (1, 1)),
        node(A, "A"),
        node(Y, "Y"),
        node(X, "X"),
        edge(X, Y, "->"),
        edge(X, A, "->"),
        edge(A, Y, "->"),
      )

      = $Y(1), Y(0) #rotate(-90deg)[$tack.r.double.not$] A$

      == Intervention is not random
      (with respect to the confounders)
    ]
  ][
    #set align(center)
    #uncover(2)[
      = RCT data
      #diagram(
        cell-size: 10mm,
        node-stroke: 0.6pt,
        spacing: 1.5em,
        let (X, A, Y) = ((0, 0), (-1, 1), (1, 1)),
        node(A, "A"),
        node(Y, "Y"),
        node(X, "X"),
        edge(X, Y, "->"),
        edge(A, Y, "->"),
      )

      = $Y(1), Y(0) tack.t.double A$

      == Force random assignment of the intervention
    ]
  ]
]

#slide(title: [Illustration: RCT data])[
  #figure(
    image("img/po_reminder_and_dags/sample_rct.svg", width: 83%),
    //caption: "DAG for a RCT: the treatment is independent of the confounders",
  )
]


#slide(title: [Illustration: RCT data, a naive solution #only(2)[that works!]])[
  == Compute the difference in mean (DM): $tau_(text("DM"))=$ $$#c_treated[$EE[Y(1)]$] - #c_control($EE[Y(0)]$)

  #figure(image("img/po_reminder_and_dags/sample_rct_mean_difference.png", width: 75%))
]

#slide(title: "Four steps of causal inference")[
  #set align(center)
  #image("img/po_reminder_and_dags/complete_inference_flow.png", width: 95%)

  #uncover(2)[
    == Today: Framing and identification with DAGs
  ]
]

#new-section-slide("Framing: How to ask a sound causal question")
//lalonde example from the book (I am less fan of no intervention examples).

/* #slide(
  title: "Identify the target trial",
)[
  What would be the ideal *randomized experiment* to answer the question?
  #cite(<hernan2016using>)
] */

#slide(title: [PICO framework @richardson1995well])[
  Originally designed for clinical research. It is a structured approach to
  formulate a research question. Critical for health technology assessment (eg. Haute Autorité de santé).

  == PICO stands for

  - Population : Who are we interested in?
  - Intervention : What treatment/intervention do we study?
  - Comparison : What are we comparing it to?
  - Outcome : What are we interested in?
  //- Define the causal measure //a bit too much for intro
  #uncover(2)[
    == Example with the job dataset @lalonde1986evaluating
    Built to evaluate the impact of the National Supported Work (NSW) program. The
    NSW is a transitional, subsidized work experience program targeted towards
    people with longstanding employment problems.
  ]
]

#slide(title: "The PICO framework")[
  #set text(size: 16pt)
  #table(
    columns: (auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header([*Component*], [*Description*], [*Example*]),
    "Population", "What is the target population of interest?", "People with longstanding employment problems",
    "Intervention", "What is the intervention?", "On-the-job training lasting between nine months and a year",
    "Control", "What is the relevant comparator?", "No training",
    "Outcome", "What are the outcomes?", "Earnings in 1978",
    "Time",
    "Is the start of follow-up aligned with intervention assignment?",
    "The period of follow-up for the earning is the year after the intervention",
  )

]

#slide(title: [PICO: other examples in econometrics])[
  The Oregon Health Insurance Experiment @finkelstein2012oregon : A randomized experiment by lottery assessing the impact of Medicaid on low-income adults in Oregon.

  - P: Low-income adults in Oregon
  - I: Medicaid
  - C: No insurance
  - O: Healthcare uses and expenditures, health outcomes
]

#slide(title: [PICO: other examples in econometrics])[
  The economic impact of climate change on US agricultural land. @deschenes2007economic: difference-in-differences design assessing the impact of climate change on agricultural profits.

  - P: US agricultural land
  - I: Climate change
  - C: No climate change
  - O: Agricultural profits
]



#slide(title: [PICO: other examples in econometrics])[
  The impact of class size on test scores. @angrist1999using: regression discontinuity design.

  - P: Fourth and fifth grades school in Israel
  - I: Class size increases by one unit
  - C: No class size increase
  - O: Test scores (math and reading)
]


#new-section-slide("Identification: List necessary information to answer the causal question")

#slide(title: [Identification: Build the causal model])[

  #quote(attribution: [@elwert2014endogenous], block: true)[
    A causal effect is said to be identified if it is possible, with ideal data (infinite sample size and no measurement error), to purge an observed association of all noncausal components such that only the causal effect of interest remains.
  ]

  == Steps
  - Potential outcome framework : mathematical tool to reason about causality

  - Directed acyclic graphs (DAG) : graphical tool to reason about causality

  - Causal estimand : what is the targeted quantity?
]

#slide(title: [Potential outcomes, @neyman1923applications @rubin1974estimating])[
  The Neyman-Rubin model, let:
  - $Y$ be the outcome,
  - $A$ the (binary) treatment

  For each individual, we have two potential outcomes: $Y(1)$ and $Y(0)$. But only
  one is observed, depending on the treatment assignment: $Y(A)$.

  #pause
  #figure(image("img/po_reminder_and_dags/sample_po.png", width: 60%))

]


#slide(title: "Causal estimand: What is the targeted quantity (with potential outcomes)?")[
  - Average treatment effect (ATE): #text(size: 1.3em)[$EE[Y(1) - Y(0)]$]
  - Conditional average treatment effect (CATE): #text(size: 1.3em)[$EE[Y(1) - Y(0) | X]$]

  #pause
  #figure(image("img/po_reminder_and_dags/po_sample_estimand.png", width: 80%))

]

#slide(title: "Causal estimand: What is the targeted quantity (with potential outcomes)?")[
  == Other estimands

  - Average treatment effect on the treated (ATT): $EE[Y(1) - Y(0) | A = 1]$
  - Conditional average treatment effect on the treated (CATT): #linebreak() #align(right)[$EE[Y(1) - Y(0) | A = 1, X]$]

  #pause
  == Other estimands more used in epidemiology

  - Risk ratio (RR): $EE[Y(1)] / EE[Y(0)]$
  - Odd ratio (OR) for binary outcome: $big(paren.l) PP[Y(1)=1] / PP[Y(1)=0] Big(paren.r) big(slash) big(paren.l)PP[Y(0)=1] / PP[Y(0)=0]big(paren.r)$

  See @colnet2023risk for a review of the different estimands and the impact on generalization.
]

#slide(title: "PICO framework, link to the potential outcomes")[
  #set text(size: 16pt)
  #table(
    columns: (auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header([*Component*], [*Description*], [*Notation*], [*Example*]),
    "Population",
    "What is the target population of interest?",
    $ X ∼ P(X) $,
    "People with longstanding employment problems",

    "Intervention",
    "What is the intervention?",
    $ A ∼ P(A = 1) = p_A $,
    "On-the-job training lasting between nine months and a year",

    "Control", "What is the relevant comparator?", $ 1 - A ∼ 1 - p_A $, "No training",
    "Outcome", "What are the outcomes?", $ Y(1), Y(0) ∼ P(Y(1), Y(0)) $, "Earnings in 1978",
    "Time",
    "Is the start of follow-up aligned with intervention assignment?",
    "N/A",
    "The period of follow-up for the earning is the year after the intervention",
  )
]


#slide(title: "Identification: assumptions")[
  == What can we learn from the data?

  - Four assumptions, referred as strong ignorability

  - Required to assure identifiability of the causal estimands with observational data @rubin2005causal

]


#slide(title: "Assumption 1: Unconfoundedness, also called ignorability")[

  == Treatment assignment is as good as random given the covariates $X$

  #eq[
    ${Y(1), Y(0)} tack.t.double A | X$
  ]
  - Equivalent to conditional independence on the propensity score: #linebreak() $e(X) \u{225D} PP(A=1|X)$ @rosenbaum1983central:

  #eq[${Y(1), Y(0)} A | e(X)$]

  #pause
  - Knowledge based ie. cannot be validated with data //Still, there is some work on causal discovery, mostly based on conditional independence tests @glymour2019review.
    - Because of possibly unmeasured confounders

    - In practice : ask yourself if you have measured all the relevant variables that could influence both the treatment and the outcome.
]



#slide(title: "Assumption 2: Overlap, also known as positivity")[

  === The treatment is not deterministic given $X$

  #align(center)[#text(size:1.4em)[$eta < e(x) < 1 - eta$] with $e(X) \u{225D} PP(A=1|X)$]

  #figure(image("img/po_reminder_and_dags/overlap.png", width: 60%))
  - Choice of covariates X: trade-off between ignorability and overlap @d2021overlap
]

#slide(title: "Assumption 3 and 4: Consistency and generalization")[

  == Consistency, also called Stable Unit Treatment Values (SUTVA)

  The observed outcome is the potential outcome of the assigned treatment for each unit i.

  #eq[$Y_(i) = A_(i) Y_(i)(1) + (1 -A_(i))Y_(i)(0)$]

  - The intervention A is well defined @hernan2020causal
  - There is no interfrence ie. network effect
  #uncover(2)[
    == Generalization, also called no-covariate shift

    Training and test data are are drawn from the same distribution
  ]
]

#new-section-slide("Directed acyclic graphs (DAGs)")

#slide(title: "Directed acyclic graphs (DAG), a tool to reason about causality")[
  == DAGs encode the causal structure of the data generating process

  Introduced by @pearl1995causal, @pearl2000models. Good practical overview in @vanderweele2019principles.

  == Motivation

  - Reason about the relation between variables.
  - Help identify for which (minimal) set of variables, the ATE is identifiable.
]

#slide(title: "Directed acyclic graphs (DAG), definitions")[
  #align(center)[
    #diagram(
      cell-size: 10mm,
      node-stroke: 0.6pt,
      spacing: 1.5em,
      let (X, A, Y) = ((0, 0), (-1, 1), (1, 1)),
      node(A, "A"),
      node(Y, "Y"),
      node(X, "X"),
      edge(X, Y, "->"),
      edge(X, A, "->"),
    )
  ]
  - #alert[Graph:] A set of relations between nodes described by edges between those nodes
  - #alert[Directed:] Edges between nodes have direction. The direction of the arrow represents a cause-effect relationship.
  - #alert[Acyclic:] : There are no cycles or loops in the causal structure. A variable can’t be a cause of itself.
]

#slide(title: "A cyclic graph")[
  #align((center))[
    == This is not a DAG
    #v(4em)
    #diagram(
      cell-size: 30mm,
      node-stroke: 2pt,
      spacing: 2em,
      let (X, A, Y) = ((0, 0), (-1, 1), (1, 1)),
      node(A, "A"),
      node(Y, "Y"),
      node(X, "X"),
      edge(X, Y, "->"),
      edge(Y, A, "->"),
      edge(A, X, "->"),
    )
  ]
]

#slide(title: "DAGs: nodes")[
  #align(center)[
    #diagram(
      cell-size: 10mm,
      node-stroke: 0.6pt,
      spacing: 1.5em,
      let (X, A, Y) = ((0, 0), (-1, 1), (1, 1)),
      node(A, "A"),
      node(Y, "Y"),
      node(X, "X"),
      edge(X, Y, "->"),
      edge(X, A, "->"),
    )
  ]

  - #alert[Nodes] represent random variables.
  - #alert[Edges] between nodes denote the presence of causal effects (i.e. difference in potential outcomes). Here, $Y_(i)(a) != Y_(i) (a')$ for two different levels of $A_(i)$ because there is an arrow from A to Y.
  - #alert[Lack of edges] between nodes denotes the absence of a causal relationships.

  #imp_block[Not drawing an arrow makes a stronger assumption about the relationship between those two variables than drawing an arrow.]
]

#slide(title: "DAGs: paths")[

  - #alert[A path] between two nodes is a route that connects the two nodes
  following non-intersecting edges.
  - A path exists even if the arrows are not pointing in the good direction.

  - Two examples of paths between A and Y:

  #grid(
    columns: (1fr, 1fr),
    gutter: 3pt,
    align: center,
    [
      $A arrow.l X arrow Y$ #linebreak()
      #diagram(
        cell-size: 10mm,
        node-stroke: 0.6pt,
        spacing: 1.5em,
        let (X, A, Y) = ((0, 0), (-1, 1), (1, 1)),
        node(A, "A"),
        node(Y, "Y"),
        node(X, "X"),
        edge(X, Y, "->"),
        edge(X, A, "->"),
      )
    ],
    [
      $A arrow Y$ #linebreak()
      #diagram(
        cell-size: 10mm,
        node-stroke: 0.6pt,
        spacing: 1.5em,
        let (X, A, Y) = ((0, 0), (-1, 1), (1, 1)),
        node(A, "A"),
        node(Y, "Y"),
        node(X, "X"),
        edge(A, Y, "->"),
      )
    ],
  )
]


#slide(title: "DAGs: causal paths")[

  - Paths encode dependencies between random variables (not necessarily causal dependecies; it can be mere associations).

  - We distinguish:
    - #alert[Causal] paths: arrows are all in the same direction.
    - From #alert[Non-causal] paths: arrows pointing in different directions
  - When there is a causal path between two variables A and B, we say that B is a #alert[descendant] of A (it is causally impacted by A)

  #grid(
    columns: (1fr, 1fr),
    gutter: 3pt,
    align: center,
    [
      #diagram(
        cell-size: 10mm,
        node-stroke: 0.6pt,
        spacing: 1.5em,
        let (X, A, Y) = ((0, 0), (-1, 1), (1, 1)),
        node(A, "A"),
        node(Y, "Y"),
        node(X, "X"),
        edge(X, Y, "->"),
        edge(X, A, "->"),
        edge(A, Y, "->"),
      )
    ],
    [
      $A arrow Y$ is #alert[causal] #linebreak()
      $Y arrow A$ is #alert[non-causal]
    ],
  )
]

#slide(title: [Your turn])[

  #set align(center)
  #diagram(
    cell-size: 10mm,
    node-stroke: 0.6pt,
    spacing: 2.5em,
    node-shape: circle,
    let (X_1, X_2, X_3, X_4, X_5) = ((0, 0), (0, 1), (1, 0), (1, 1), (-1, 2)),
    node(X_1, $X_1$),
    node(X_2, $X_2$),
    node(X_3, $X_3$),
    node(X_4, $X_4$),
    node(X_5, $X_5$),
    edge(X_1, X_2, "->"),
    edge(X_1, X_3, "->"),
    edge(X_2, X_4, "->"),
    edge(X_2, X_4, "->"),
    edge(X_5, X_1, "->"),
    edge(X_5, X_4, "->"),
  )

  Paths between $X_1$ and $X_4$? Which of them are causal?

  #uncover(2)[
    - $X_5 arrow X_1 arrow X_2 arrow X_4$
    - $X_5 arrow X_4$
    - $X_1 arrow.l X_5 arrow X_4$ : non causal
    - $X_3 arrow.l X_1 arrow X_2 arrow X_4$ : non causal
  ]
]

#slide(title: "Three types of directed edges: path")[


  There are three kinds of “triples" or paths with three nodes: These constitute the most basic building blocks for causal DAGs.


  == First, a #alert[causal] path (or chain): $A arrow B arrow C$

  #align(center)[
    #diagram(
      cell-size: 10mm,
      node-stroke: 0.6pt,
      spacing: 1.5em,
      node-shape: circle,
      let (A, B, C) = ((-1, 1), (0, 0), (1, 1)),
      node(A, $A$),
      node(B, $B$),
      node(C, $C$),
      edge(A, B, "->"),
      edge(B, C, "->"),
    )
  ]

  #only(1)[
    - The effect of A “flows" through B. A and C are not independent and the relationship is causal.
  ]

  #only(2)[
    #ex_block("An individual receiving a message (A) encouraging them to vote causes that individual to actually vote (C) only if the individual actually reads (B) the message.")
  ]
]

#slide(title: "Three types of directed edges: mutual depence")[

  == Second, #alert[mutual dependence] or fork or confounder: $A arrow.l B arrow C$

  #align(center)[
    #diagram(
      cell-size: 10mm,
      node-stroke: 0.6pt,
      spacing: 1em,
      node-shape: circle,
      let (A, B, C) = ((-1, 1), (0, 0), (1, 1)),
      node(A, $A$),
      node(B, $B$),
      node(C, $C$),
      edge(B, A, "->"),
      edge(B, C, "->"),
    )
  ]

  - A and C are not causally related but B is a common cause of both.
  - A and C are not independent, but are independent conditional on B.
  - #alert[Not conditionning on X] introduces bias.

  #ex_block(title: "")[
    - Rise in temperature (B) causes both the thermometer (A) to change, and ice to melt (C), but the thermometer changing does not cause ice to melt.
    - A is prostate cancer; B is age; and C is Alzheimer’s disease.
  ]
]

#slide(title: "Three types of directed edges: collider")[

  == Third, #alert[collider]: $A arrow B arrow.l C$

  #align(center)[
    #diagram(
      cell-size: 10mm,
      node-stroke: 0.6pt,
      spacing: 1.5em,
      node-shape: circle,
      let (A, B, C) = ((-1, 1), (0, 0), (1, 1)),
      node(A, $A$),
      node(B, $B$),
      node(C, $C$),
      edge(A, B, "->"),
      edge(C, B, "->"),
    )
  ]

  - A and C are both common causes of B : they collide into B.
  - A and C are independent, but *conditionnaly* dependent given B.
  - #alert[Conditionning on B] introduces a spurious correlation between A and C.

  #ex_block(title: "")[
    - A is result from dice 1, C is results from dice 2, B is sum of dice 1 and dice 2.
    - A is height, C is speed, B is whether an athlete plays in the NBA.
  ]
]


#slide(title: "Open and blocked paths ")[

  We can open or block paths between variables by conditionning on them:

  #align(center)[
    #diagram(
      cell-size: 5mm,
      node-stroke: 0.6pt,
      spacing: 1em,
      node-shape: circle,
      let (A, X, Y) = ((-1, 1), (0, 0), (1, 1)),
      node(A, $A$),
      node(X, $X$),
      node(Y, $Y$),
      edge(X, A, "->"),
      edge(X, Y, "->"),
      edge(A, Y, "->"),
    )
  ]
  A path is #alert[blocked] (or d-separated) if:
  - the path contains a non-collider that has been conditioned on.
  - or the path contains a collider that has not been conditioned on (and has no descendants that have been conditioned on).

  Conditioning on a variable:
  - #alert[Blocks] a path if that variable is #alert[not a collider] on that path.
  - #alert[Opens] a path if that variable is a #alert[collider] on that path.
]


#slide(title: [Your turn])[

  #set align(center)
  #diagram(
    cell-size: 10mm,
    node-stroke: 0.6pt,
    spacing: 2.5em,
    node-shape: circle,
    let (X_1, X_2, X_3, X_4, X_5) = ((0, 0), (0, 1), (1, 0), (1, 1), (-1, 2)),
    node(X_1, $X_1$),
    node(X_2, $X_2$),
    node(X_3, $X_3$),
    node(X_4, $X_4$),
    node(X_5, $X_5$),
    edge(X_1, X_2, "->"),
    edge(X_1, X_3, "->"),
    edge(X_2, X_4, "->"),
    edge(X_2, X_4, "->"),
    edge(X_5, X_1, "->"),
    edge(X_5, X_4, "->"),
  )

  Paths from $X_5$ to $X_2$? Which of them are opened/blocked?

  #pause
  $X_5 arrow X_1 arrow X_2$ (blocked by conditionning on $X_1$)

  $X_5 arrow X_1 arrow X_2 arrow X_4$ (opened by conditionning $X_4$)

]



#slide(title: "Backdoor paths: a special type of paths")[


  #imp_block[
    Backdoor path: A Backdoor path from a variable A to another Y, is any non-causal path between A and Y that does not include descendants of A
  ]

  #alert[Identifying backdoor paths:] Backdoor paths from A to Y are all those paths that remain between A and Y after removing all arrows coming out of A.

  These paths are responsible for confounding bias: they imply association not causation.

  #grid(
    columns: (1fr, 1fr),
    gutter: 3pt,
    align: center,
    align(center)[
      #diagram(
        cell-size: 5mm,
        node-stroke: 0.6pt,
        spacing: 1em,
        node-shape: circle,
        let (A, X, Y) = ((-1, 1), (0, 0), (1, 1)),
        node(A, $A$),
        node(X, $X$),
        node(Y, $Y$),
        edge(X, A, "->"),
        edge(X, Y, "->"),
        edge(A, Y, "->"),
      )
    ],
    align(center)[
      #diagram(
        cell-size: 5mm,
        node-stroke: 0.6pt,
        spacing: 1em,
        node-shape: circle,
        let (A, X, Y) = ((-1, 1), (0, 0), (1, 1)),
        node(A, $A$),
        node(X, $X$),
        node(Y, $Y$),
        edge(X, A, "->"),
        edge(X, Y, "->"),
      )
    ],
  )
]


#slide(title: [Your turn])[

  #set align(center)
  #diagram(
    cell-size: 10mm,
    node-stroke: 0.6pt,
    spacing: 2.5em,
    node-shape: circle,
    let (X_1, X_2, X_3, X_4, X_5) = ((0, 0), (0, 1), (1, 0), (1, 1), (-1, 2)),
    node(X_1, $X_1$),
    node(X_2, $X_2$),
    node(X_3, $X_3$),
    node(X_4, $X_4$),
    node(X_5, $X_5$),
    edge(X_1, X_2, "->"),
    edge(X_1, X_3, "->"),
    edge(X_2, X_4, "->"),
    edge(X_2, X_4, "->"),
    edge(X_5, X_1, "->"),
    edge(X_5, X_4, "->"),
  )

  What are the backdoor paths from $X_1$ to $X_4$? from $X_2$ tot $X_4$?

  #pause
  $X_1 arrow.l X_5 arrow X_4$

  $X_2 arrow.l arrow.l X_5 arrow X_4$


]


#slide(title: "Graphical identification")[

  DAGs help us know whether observed covariates are enough to
  identify a treatment effect.

  #align(center)[
    #diagram(
      cell-size: 5mm,
      node-stroke: 0.6pt,
      spacing: 1em,
      node-shape: circle,
      let (A, X, Y) = ((-1, 1), (0, 0), (1, 1)),
      node(A, $A$),
      node(X, $X$),
      node(Y, $Y$),
      edge(X, A, "->"),
      edge(X, Y, "->"),
      edge(A, Y, "->"),
    )
  ]

  In other words, how can we make it so that there are no non-causal
  dependencies between treatment and outcome?

  #imp_block(title: [Graphical identification @pearl2000models])[
    The effect of A on Y is identified if all backdoor paths from A to B are blocked, and no descendant of A is conditioned on.
  ]
]

#slide(title: "On which variables should we condition? General rules")[

  - Do not condition for variables on causal paths from treatment to outcome
  - Condition on variables that block non-causal backdoor paths
  - Don’t condition on colliders! Eg. don't condition on post-treatment variables.

  #pause
  In the following example, to estimate the effect of T on Y, we should:
  - Condition on X
  - NOT condition on M because it is a descendant of T

  #align(center)[
    #diagram(
      cell-size: 5mm,
      node-stroke: 0.6pt,
      spacing: 1em,
      node-shape: circle,
      let (A, X, Y, C) = ((-1, 1), (0, 0), (1, 1), (0, 2)),
      node(A, $A$),
      node(X, $X$),
      node(Y, $Y$),
      node(C, $C$),
      edge(X, A, "->"),
      edge(X, Y, "->"),
      edge(A, Y, "->"),
      edge(A, C, "->"),
      edge(Y, C, "->"),
    )
  ]

]


#slide(title: "Famous examples of confounders")[

  #figure(image("img/po_reminder_and_dags/confounder.svg", width: 20%))

  #ex_block(title: [Effect of education on earnings])[
    The family background can act as a confounder: Wealthier families may provide better education opportunities AND influence earnings independently of the education itself.
  ]
]



#slide(title: "Famous examples of instrumental variables")[
  === Instrumental variable (IV): influences only the treatment.

  #figure(image("img/po_reminder_and_dags/instrumental_variable.svg", width: 20%))

  #ex_block(title: [Effect of education on earnings @angrist1991does])[
    Quarter of birth are randomly assigned but influence the lengths of schooling due to school entry laws.
  ]
]


#slide(title: "Famous examples of colliders")[

  #figure(
    image("img/po_reminder_and_dags/collider_pre_outcome.excalidraw.svg", width: 20%),
    caption: "Special case of collider: consequence of both the treatment and the outcome.",
  )

  #ex_block(title: [Effect of smoking on mortality @hernandez2006birth])[
    Birth weight is influenced by smoking and other factors. Conditioning on birth weight (a collider) creates a spurious negative correlation between smoking and other risk factors, leading to the paradoxical conclusion that smoking reduces infant mortality, even though it harms overall health.
  ]
]

#slide(title: "More colliders: M-bias")[

  #align(center)[
    #diagram(
      cell-size: 5mm,
      node-stroke: 0.6pt,
      spacing: 0.8em,
      node-shape: circle,
      let (A, X, Y, U_1, U_2) = ((-1, 1), (0, 0), (1, 1), (-1, -1), (1, -1)),
      node(A, $A$),
      node(X, $X$),
      node(Y, $Y$),
      node(U_1, $U_1$, stroke: (dash: "dashed")),
      node(U_2, $U_2$, stroke: (dash: "dashed")),
      edge(A, Y, "->"),
      edge(U_1, X, "-->"),
      edge(U_2, X, "-->"),
      edge(U_1, A, "-->"),
      edge(U_2, Y, "-->"),
    )
  ]
  - Do not condition on any pre-exposure variable that you have at disposal!

  - Should we condition on X in trying to estimate the effect of A on Y ?

  - There is a backdoor path through two unobserved variables $(U_1 , U_2)$. But it is blocked because X is a collider along that path.

  - Conditioning on X opens up that path, inducing a non-causal association between T and Y.
]


#slide(title: "Which variable to include into your analysis?")[

  #grid(
    columns: (auto, auto, auto, auto),
    gutter: 1pt,
    figure(
      image("img/po_reminder_and_dags/confounder.svg", width: 40%),
      caption: [Confounder ✅],
    ),
    figure(
      image("img/po_reminder_and_dags/instrumental_variable.svg", width: 40%),
      caption: [Instrumental variable ❌ #linebreak() (generally)],
    ),
    figure(
      image("img/po_reminder_and_dags/dag_x_to_y.svg", width: 40%),
      caption: [Outcome parent ✅ #linebreak() (generally)],
    ),
    figure(
      image("img/po_reminder_and_dags/collider.svg", width: 40%),
      caption: [Collider ❌],
    ),
  )

  - High-level strategy: Control solely for pre-treatment variables that influences both the outcomes, the treatment or both.
  /* #set align(bottom)

  #grid(
    columns: (auto, auto),
    gutter: 1pt,
    figure(
      image("img/po_reminder_and_dags/mediator.svg", width: 40%),
    caption: [Mediator #uncover(2)[❌ (generally)]],
    ),
    figure(
      image("img/po_reminder_and_dags/effect_modifier.svg", width: 40%),
    caption: [Effect modifier #uncover(2)[✅ (generally)]],
    ),
    ) */
]

#slide(title: "Special types of variables: mediators")[
  A #alert[mediator] block the path from the treatment to the outcome.
  #side-by-side(
    [
      #align(center)[
        #diagram(
          cell-size: 5mm,
          node-stroke: 0.6pt,
          spacing: 1em,
          node-shape: circle,
          let (A, M, Y) = ((-1, 1), (0, 0), (1, 1)),
          node(A, $A$),
          node(M, $M$),
          node(Y, $Y$),
          edge(A, M, "->"),
          edge(M, Y, "->"),
          edge(A, Y, "->"),
        )
      ]
    ],
    [
      Here, two causal paths from A:
      - $A arrow Y$ - a “direct effect"
      - $A arrow M arrow Y$ - an “indirect effect" through M
    ],
  )
  #only(2)[
    - All causal paths from a treatment capture its overall treatment effect.

    - The average treatment effect of T combines both the “direct effect" and the “indirect effect".
  ]

  #only(1)[
    #ex_block(title: [Effect of children poverty on economic outcomes @bellani2019long])[
      - Y is economic outcomes in adulthood, A is child poverty, M is education.
      - What part of the effect of poverty on outcome is mediated by education?
    ]
  ]
]

#slide(title: "DAG: Effect modifier")[
  === Effect modifier: influences the treatment effect on the outcome.

  #figure(image("img/po_reminder_and_dags/effect_modifier.svg", width: 20%))
]


#slide(title: "Take away on DAGs")[
  - DAGs are a powerful tool to reason about causality
  - Useful to identify the variables to condition on / to include into the analysis
  - NB: Drawing the true DAG is often hard / not feasable
]


#slide(title: [Take away on covariate selection])[

  === What is the most important part to insure validity?

  - The covariate included (an appropriate DAG)?

  - The design of the study?

  - The causal estimator (IPW, G-formula, AIPW...)

  - The statistical estimator (Linear regression, Logistic regression...)
]

#slide(title: [Comparing different choices for the study steps @doutreligne2024causalthinking])[
  == Question of interest
  #figure(image("img/po_reminder_and_dags/applied_inference_flow.png", width: 100%))
]


#slide(title: "Results of the sensitivity analysis")[
  #figure(image("img/po_reminder_and_dags/tutorial_ci_results.png", width: 50%))
]

#new-section-slide("Practical session")

#slide(title: "To your notebooks! 👩‍💻")[
  - url: https://straymat.github.io/causal-ml-course/practical_sessions.html
]

#let bibliography = bibliography("biblio.bib", style: "apa")

#set align(left)
#set text(size: 20pt, style: "italic")
#slide[
  #bibliography
]


#new-section-slide("Supplementary material")


#slide(title: "A word on structural equation models")[
  TODO
]
