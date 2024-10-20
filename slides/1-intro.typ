// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.3.1": *

#import themes.metropolis: *

#show: metropolis-theme.with(
    footer: [ENSAE, Introduction course]
)

// Make the paper dimensions fit for a presentation and the text larger
#set page(paper: "presentation-16-9")
#set text(size: 22pt)


// Use #polylux-slide to create a slide and style it using your favourite Typst functions
// #polylux-slide[
//   #align(horizon + center)[
//     = Very minimalist slides

//     A lazy author

//     July 23, 2023
//   ]
// ]

#title-slide(
  author: [Authors],
  title: "Machine Learning for econometrics",
  subtitle: "Causal perspective",
  date: "January 10, 2025",
  //extra: "Extra"
)

#slide(title: "Table of contents")[
  #metropolis-outline
]

#new-section-slide("Introduction")

#slide()[
  Causal inference: subfield of statistics dealing with "why questions".
  #set align(horizon + center)
  #image("img/intro/confounder.png", width: 15%)
  
  #set align(horizon + left)

  At the center of epidemiology, econometrics, social sciences...  
  
  // Make use of features like #uncover, #only, and others to create dynamic content
  #uncover(2)[Now, bridging with Machine Learning @kaddour2022causal]
  
]

#new-section-slide("Asking a sound causal question: PICO framework")

#slide[
  == Identify the target trial
  What would be the ideal *randomized experiment* to answer the question?
  #cite(<hernan2016using>)

]

#slide[
  == PICO framework

  - Population : Who are we interested in?
  - Intervention : What treatment/intervention do we study?
  - Comparison : What are we comparing it to?
  - Outcome : What are we interested in?
]


#new-section-slide("Causal graphs")

#new-section-slide("How to ask a sound causal question")

#slide[
  == What is a #alert[why question] ?

- Economics: How does supply and demand (causally) depend on price?

- Policy: Are job training programmes actually effective?

- Epidemiology: How does this threatment affect the patient's health? 

- Public health : Is this prevention campaign effective?

- Psychology: What is the effect of family structure on children’s outcome?

- Sociology: What is the effect of social media on political opinions?
]

#slide[
  == This is different from a #alert[predictive question]

  - What will be the weather tomorrow?
  - What will be the outcome of the next election?
  - How many people will get infected by flue next season?
  - What is the cardio-vacular risk of this patient?
  - How much will the price of a stock be tomorrow?
]

#slide[
  == Why is #alert[prediction different from causation]?


  - Prediction assumes stability between train and test data
  #uncover(2)[- Causal inference search the effect of an intervention]
]


#slide[
  == How to ask a sound causal question

  - Define the population of interest
  - Define the intervention
  - Define the outcome
  - Define the counterfactual
  - Define the causal effect
]


#new-section-slide("Different steps : identification, estimation, inference")

#slide[
  Identification: what can we learn from the data?
]

#slide[
  Identification: what can we learn from the data?
]

#new-section-slide("Causal graphs")

#new-section-slide("Potential outcomes")

#new-section-slide("Related concepts")

#set align(horizon + center)

Structural equations.


#slide[
  #set align(start + top)

  Hello world
]

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