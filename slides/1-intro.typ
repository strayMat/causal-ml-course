// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.3.1": *

#import themes.metropolis: *

#show: metropolis-theme.with(
    footer: [ENSAE, Introduction course]
)

// Make the paper dimensions fit for a presentation and the text larger
#set page(paper: "presentation-16-9")
#set text(size: 25pt)


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

  At the center of epidemiology, econometrics, social sciences. Now, bridging with ML.

  #set align(horizon + center)
  #image("img/intro/confounder.png", width: 15%)
  
]

#slide[
  == This slide changes!

  You can always see this.
  // Make use of features like #uncover, #only, and others to create dynamic content
  #uncover(2)[But this appears later!]
]

#focus-slide[
  Wake up!
]


#new-section-slide("Asking a sound causal question: PICO framework")


#slide[
  == Identify the target trial

  What would be the ideal *randomized experiment* to answer the question?

  #uncover(2)[#cite(<hernan2016using>)]
]

#slide[
  == PICO framework

  - Population : Who are we interested in?
  - Intervention : What treatment/intervention do we study?
  - Comparison : What are we comparing it to?
  - Outcome : What are we interested in?
]


#new-section-slide("Causal graphs")

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


#slide[
  #bibliography("biblio.bib", style: "harvard-cite-them-right")
]