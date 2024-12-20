// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.3.1": *
#import "@preview/embiggen:0.0.1": * // LaTeX-like delimiter sizing for Typst
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
  subtitle: "Flexible models for tabular data",
  date: "February 18th, 2025",
  //extra: "Extra"
)

#slide(title: "Reminder from previous session")[

- Statistical learning 101: bias-variance trade-off

- Regularization for linear models: Lasso, Ridge, Elastic Net

- Transformation of variables: polynomial regression

#uncover(2)[- 🤔 But... How to select the best model? the best hyperparameters?]
]


#slide(title: "Table of contents")[
  #metropolis-outline
]

]
#new-section-slide("Model evaluation and selection with cross-validation")

#slide(title: "Model evaluation: example")[

  - We saw the importance to split the data into training and testing sets. 


]

#new-section-slide("Tree, random forests and boosting")

#slide(title:"Random Forests for predictive inference")[

]

#slide(title:"Boosting")[

]

#slide(title:"Ensemble models")[]

#new-section-slide("A word on other families of models")

#slide(title: "Why not use deep learning everywhere?")[

- Success of deep learning in image, speech recognition and text

- Why not so used in economics?
]


#slide(title: "Limited data settings")[
  - Typically #only(1)[in economics] #only(1)[everywhere], we have a limited number of observations

  #figure(
    image("img/ML_1/2020_kdd_dataset_sizes.png", width: 65%),
    caption: [Typical dataset are mid-sized. This does not change with time. #footnote("https://www.kdnuggets.com/2020/07/poll-largest-dataset-analyzed-results.html")],
  )

]

#slide(title: "Deep learning underperforms on data tables")[

 == Tree-based methods outperform tailored deep learning architectures @grinsztajn2022tree

 #figure(
    image("img/ML_1/tree_outperforms_dl.png", width: 83%),
    caption: "DAG for a RCT: the treatment is independent of the confounders",
  )
]


#slide(title: "Other well known families of models")[

  - Generalized linear models: 

  - Support vector machines:

  - Gaussian processes: 
  
]



#let bibliography = bibliography("biblio.bib", style: "apa")

#set align(left)
#set text(size: 20pt, style: "italic")
#slide[
  #bibliography
]