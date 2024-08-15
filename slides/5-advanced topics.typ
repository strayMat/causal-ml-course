// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.3.1": *

#import themes.metropolis: *

#show: metropolis-theme.with(
    footer: [Custom footer]
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
  title: "Causal Machine Learning",
  subtitle: "Introduction",
  date: "January 10, 2025",
  //extra: "Extra"
)

#slide(title: "Table of contents")[
  #metropolis-outline
]

#new-section-slide("Introduction")

#slide()[
  Some inspiratong stuff
]

#new-section-slide("Synthetic Controls")

#slide()[
  == Synthetic Controls
  Introduced in @abadie2003economic and @abadie2010synthetic, well described in @abadie2021using
  - A method for estimating the effect of a treatment on a single unit
  - The treatment unit is compared to a weighted average of control units
  - The weights are chosen to minimize the difference between the treated unit and the synthetic control
  
  Example for the effect of taxes on sugar-based product consumption in @puig2021impact, review of usage in healthcare @bouttell2018synthetic.
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

#slide[
  == Resources
  
- https://web.stanford.edu/~swager/stats361.pdf
- https://www.mixtapesessions.io/
- https://alejandroschuler.github.io/mci/
]


#slide[
  == Bibliography
  
#bibliography("biblio.bib")  
]
