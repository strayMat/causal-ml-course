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

#slide()[
  Causal inference: subfield of statistics dealing with "why questions".
  #set align(horizon + center)
  #image("img/intro/confounder.png", width: 15%)
  
  #set align(horizon + left)

  At the center of epidemiology, econometrics, social sciences...  
  
  // Make use of features like #uncover, #only, and others to create dynamic content
  #uncover(2)[Now, bridging with Machine Learning @kaddour2022causal]
  
]

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


  - Prediction (most part of Machine Learning) focus on understanding what usually happens in a given situation.
  
  #uncover(2)[It assumes iid between train and test data.]

  #uncover(3)[
  
  - Causal inference (most part of economists) focus on what would happen if we changed the system ie. under intervention.
  
  It models the covariate shift between treated and control units.  
  ]
]

//TODO: insert images of covariate shifts to illustrate. 

#new-section-slide("How to ask a sound causal question: The PICO framework")

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
  //- Define the causal measure //a bit too much for intro 
]


#slide[
  == PICO framework (illustration)

  - P 
  - I 
  - C 
  - O 
]

#new-section-slide("Causal graphs")


#new-section-slide("The four steps of causal inferenceidentification, statistical estimand, statistical inference")


#slide[
  = Causal estimand
  
  What can we learn from the data?
]

#slide[
  = Identification

  What can we learn from the data?

  Knowledge based

  Cannot be validated with data 
]

#new-section-slide("Potential outcomes")

#new-section-slide("Causal estimands")

#new-section-slide("Causal graphs")

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