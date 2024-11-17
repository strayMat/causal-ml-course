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

  #grid(
    columns: (auto, auto),
    gutter: 3pt,
    image("img/intro/what_if.jpg", width: 40%),
    image("img/intro/causal_ml_book.png", width: 47%),
  )
  At the center of epidemiology @hernan2020causal, econometrics #cite(<chernozhukov2024applied>),
  social sciences, #only(2)[machine learning...]

  #only(
    2,
  )[
    Now, bridging with machine learning @kaddour2022causal : Fairness, reinforcement
    learning, causal discovery, causal inference for LLM, causal representations... //#footnote[Eg. #link("https://nips.cc/virtual/2024/events/workshop")[neurips 2024 workshops]]
  ]
  #only(3)[
    #alert[This course:]
    - Basis of causal inference using ML approaches (semi-parametric),
    - Inspiration from epidemiology,
    - Application in econometrics.
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
  === Prediction (most part of ML): What usually happens in a given situation?
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
  title: "Machine learning is (basically) pattern matching",
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


#slide(
  title: "Machine learning is great for prediction",
)[
  == Leverages complex data structures
  #uncover(
    1,
  )[
    - Images: Image classification with deep convolutional neural networks @krizhevsky2012imagenet
  ]
  #uncover(
    2,
  )[
    - Speech-to-text: Towards end-to-end speech recognition with recurrent neural networks @graves2014towards
  ]
  
  #uncover(
    3,
  )[
    - Text: Attention is all you need @vaswani2017attention
  ]
]

#slide(
  title: "Machine learning might be less successful for what if questions",
)[
  == Machine learning is not driven by causal mechanisms
  
  - For example people that go to the hospital die more than people who do not #footnote[Example from #link("sklearn mooc")[https://inria.github.io/scikit-learn-mooc/concluding_remarks.html?highlight=causality]]:

    - Naive data analysis might conclude that hospitals are bad for health.

    The fallacy is that we are comparing different populations: people who go to the hospital typically have a worse baseline health than people who do not.

  This is a confounding factor: a variable that influences both the treatment and the outcome.
]

#slide(
  title: "Why is prediction different from causation? (2/2)",
)[
  === Causal inference (most part of economists) : What would happen if we changed the system ie. under an intervention?
  #uncover(
    2,
  )[
    #alert([Assumption]): No unmeasured variables influencing both treatment and
    outcome $arrow$ confounders.
    #side-by-side(gutter: 3mm, columns: (1fr, 2fr))[
      #figure(image("img/intro/confounder.svg", width: 60%))
    ][
      Causal inference models #linebreak()
      $(X, A, Y(A=1), Y(A=0))$ #linebreak()
      the covariate shift between treated and control units.
    ]
  ]
]


#slide(title: [Illustration of the fundamental problem of causal inference])[
  Consider an example from epidemiology:
  - Population: patients experiencing a stroke 
  - #c_treated[Intervention $A = 1$: patients had access to a MRI scan  #text(weight: "extrabold")[in less than 3 hours] after the first symptoms]
  - #c_control[Comparator $A = 0$: patients had access to a MRI scan #text(weight: "extrabold")[in more than 3 hours] after the first symptoms]
  - $Y = PP[text("Mortality")]$: the mortality at 7 days 
  - $X = PP[text("Charlson score")]$: a comorbidity index summarizing the overall health state of the patient. Higher is bad for the patient.
]


#slide(title: [Illustration: observational data])[
  == Draw a population sample without treatment status
  #figure(
    image("img/intro/sample_wo_oracles_gray.png", width: 83%),
    //caption: "DAG for a RCT: the treatment is independent of the confounders",
  )
]

#slide(title: [Illustration: observational data])[
  == Draw a population sample with treatment status
  #figure(
    image("img/intro/sample.svg", width: 83%),
    //caption: "DAG for a RCT: the treatment is independent of the confounders",
  )
]



#slide(title: [Illustration: observational data, a naive solution])[
  == Compute the difference in mean (DM):  $tau_(text("DM"))=$ $$#c_treated[$EE[Y(1)]$] - #c_control($EE[Y(0)]$)

  #uncover(2)[
  #figure(
    image("img/intro/sample_mean_difference.png", width: 83%),
    //caption: "DAG for a RCT: the treatment is independent of the confounders",
  )
  ]
]

#slide(title: [RCT case: No problem of confounding])[
 
 #side-by-side(gutter: 10mm, columns: (1fr, 1fr))[
      #set align(center)
      = Observational data  
      #figure(
        image("img/intro/confounder.svg", width: 50%),
        )
        
        = $Y(1), Y(0) #rotate(-90deg)[$tack.r.double.not$] A$

        == Intervention is not random 
        (with respect to the confounders)
        
    ][
      #set align(center)
      #uncover(2)[
        = RCT data 
        #figure(
       image("img/intro/rct_dag.excalidraw.svg", width: 58%),
      )
      = $Y(1), Y(0) tack.t.double A$
      
      == Force random assignment of the intervention
    ] 
    ] 
]

#slide(title: [Illustration: RCT data])[
  #figure(
    image("img/intro/sample_rct.svg", width: 83%),
    //caption: "DAG for a RCT: the treatment is independent of the confounders",
  )
]


#slide(title: [Illustration: RCT data, a naive solution #only(2)[that works!]])[
  == Compute the difference in mean (DM):  $tau_(text("DM"))=$ $$#c_treated[$EE[Y(1)]$] - #c_control($EE[Y(0)]$)

  #figure(
    image("img/intro/sample_rct_mean_difference.png", width: 83%),
  )
]


#new-section-slide(
  "Four steps of causal inference : Framing, identification, statistical inference, vibration analysis",
)

#slide(title: "Complete inference flow")[
  #set align(center)
  #image("img/intro/complete_inference_flow.png", width: 95%)

  == Today: Framing and identification
]



#new-section-slide("Framing: How to ask a sound causal question")
//lalonde example from the book (I am less fan of no intervention examples).

/* #slide(
  title: "Identify the target trial",
)[
  What would be the ideal *randomized experiment* to answer the question?
  #cite(<hernan2016using>)
] */

#slide(
  title: [PICO framework @richardson1995well],
)[
  Originally designed for clinical research. It is a structured approach to
  formulate a research question. Critical for health technology assessment (eg. Haute Autorité de santé). 
  
  == PICO stands for

  - Population : Who are we interested in?
  - Intervention : What treatment/intervention do we study?
  - Comparison : What are we comparing it to?
  - Outcome : What are we interested in?
  //- Define the causal measure //a bit too much for intro
  #uncover(
    2,
  )[
    == Example with the job dataset @lalonde1986evaluating
    Built to evaluate the impact of the National Supported Work (NSW) program. The
    NSW is a transitional, subsidized work experience program targeted towards
    people with longstanding employment problems.
  ]
]

#slide(
  title: "The PICO framework",
)[
  #set text(size: 16pt)
  #table(
    columns: (auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header([*Component*], [*Description*], [*Example*]),
    "Population",
    "What is the target population of interest?",
    "People with longstanding employment problems",
    "Intervention",
    "What is the intervention?",
    "On-the-job training lasting between nine months and a year",
    "Control",
    "What is the relevant comparator?",
    "No training",
    "Outcome",
    "What are the outcomes?",
    "Earnings in 1978",
    "Time",
    "Is the start of follow-up aligned with intervention assignment?",
    "The period of follow-up for the earning is the year after the intervention",
  )

]
//TODO: what could go wrong : selection bias, and other design issues



#slide(
  title: [PICO: other examples in econometrics]
)[
  The Oregon Health Insurance Experiment @finkelstein2012oregon : A randomized experiment by lottery assessing the impact of Medicaid on low-income adults in Oregon.

  - P: Low-income adults in Oregon
  - I: Medicaid
  - C: No insurance
  - O: Healthcare uses and expenditures, health outcomes
]

#slide(
  title: [PICO: other examples in econometrics]
)[
  The economic impact of climate change on US agricultural land. @deschenes2007economic: difference-in-differences design assessing the impact of climate change on agricultural profits. 

  - P: US agricultural land
  - I: Climate change
  - C: No climate change
  - O: Agricultural profits
]



#slide(
  title: [PICO: other examples in econometrics]
)[
  The impact of class size on test scores. @angrist1999using: regression discontinuity design. 

  - P: Fourth and fifth grades school in Israel
  - I: Class size increases by one unit 
  - C: No class size increase
  - O: Test scores (math and reading)
]


#new-section-slide("Identification: List necessary information to answer the causal question")

#slide(
  title: [Build a causal model],
)[
  The identification step builds a causal model to answer the research question. 
  
  Several important steps:

  - Potential outcome framework : mathematical tool to reason about causality

  - Directed acyclic graphs (DAG) : graphical tool to reason about causality

  - Causal estimand : what is the targeted quantity?
]

#slide(
  title: [Potential outcomes, @neyman1923applications @rubin1974estimating],
)[
  The Neyman-Rubin model, let:
  - $Y$ be the outcome,
  - $A$ the (binary) treatment

  For each individual, we have two potential outcomes: $Y(1)$ and $Y(0)$. But only
  one is observed, depending on the treatment assignment: $Y(A)$.

  #uncover(2)[
   #figure(
    image("img/intro/sample_po.png", width: 70%),
  )
  ]
]


#slide(
  title: "Causal estimand: What is the targeted quantity (with potential outcomes)?",
)[
    - Average treatment effect (ATE): #text(size: 1.3em)[$EE[Y(1) - Y(0)]$]
    - Conditional average treatment effect (CATE): #text(size: 1.3em)[$EE[Y(1) - Y(0) | X]$]

  #uncover(2)[
    #figure(
    image("img/intro/po_sample_estimand.png", width: 80%),
  )
  ]
]
  
#slide(title: "Causal estimand: What is the targeted quantity (with potential outcomes)?",
)[
  == Other estimands 

  - Average treatment effect on the treated (ATT): $EE[Y(1) - Y(0) | A = 1]$
  - Conditional average treatment effect on the treated (CATT): #linebreak() #align(right)[$EE[Y(1) - Y(0) | A = 1, X]$]
  
  #uncover(
    2,
  )[
    == Other estimands more used in epidemiology

    - Risk ratio (RR): $EE[Y(1)] / EE[Y(0)]$
    - Odd ratio (OR) for binary outcome: $big(paren.l) PP[Y(1)=1] / PP[Y(1)=0] Big(paren.r) big(slash) big(paren.l)PP[Y(0)=1] / PP[Y(0)=0]big(paren.r)$

    See @colnet2023risk for a review of the different estimands and the impact on generalization.]
]

#slide(
  title: "PICO framework, link to the potential outcomes",
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
    "People with longstanding employment problems",
    "Intervention",
    "What is the intervention?",
    $ A ∼ P(A = 1) = p_A $,
    "On-the-job training lasting between nine months and a year",
    "Control",
    "What is the relevant comparator?",
    $ 1 - A ∼ 1 - p_A $,
    "No training",
    "Outcome",
    "What are the outcomes?",
    $ Y(1), Y(0) ∼ P(Y(1), Y(0)) $,
    "Earnings in 1978",
    "Time",
    "Is the start of follow-up aligned with intervention assignment?",
    "N/A",
    "The period of follow-up for the earning is the year after the intervention",
  )
]


#slide(
  title: "Identification: assumptions",
)[
 == What can we learn from the data?
 
 - Four assumptions, referred as strong ignorability
 
 - Required to assure identifiability of the causal estimands with observational data @rubin2005causal

]


#slide(title: "Assumption 1: Unconfoundedness, also called ignorability")[

== Treatment assignment is as good as random given the covariates $X$

#eq[
${Y(1), Y(0)} tack.t.double A | X$
]
- Equivalent to the conditional independence on the propensity score $e(X) hat(=) PP(A=1|X)$ @rosenbaum1983central:

#eq[${Y(1), Y(0)} A | e(X)$]

#uncover(2)[- Knowledge based ie. cannot be validated with data //#uncover(2)[Still, there is some work on causal discovery, mostly based on conditional independence tests @glymour2019review.
- Because of possibly unmeasured confounders

- In practice : ask yourself if you have measured all the relevant variables that could influence both the treatment and the outcome.
]
]


#slide(title: "Assumption 2: Overlap, also known as positivity")[

== The treatment is not deterministic given $X$

#align(center)[#text(size:1.4em)[$eta < e(x) < 1 - eta$] with $e(X) hat(=) PP(A=1|X)$] 

#figure(
  image("img/intro/overlap.png", width: 60%),
)
- NB: The choices of covariates X can be viewed as a trade-off between ignorability and overlap @d2021overlap
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

#slide(title: "Directed acyclic graphs (DAG)")[
  === A tool to reason about causality: What are the causal status of each variable?

  The confounder: a variable that influences both the treatment and the outcome.

  #uncover(2)[
    #figure(
      image("img/intro/confounder.svg", width: 30%),
    )
  ]
]



#slide(title: "A word on structural equation models")[
]


#new-section-slide("Session summary")



#new-section-slide("Going further")

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