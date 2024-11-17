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

#new-section-slide("Recall from the previous session")


#new-section-slide("Identification strategies")

#slide(title: "Identification strategies")[

]

#slide(title: "Outcome based identification")[

]


#slide(title: "Treatment based identification")[

]


#new-section-slide("Identification proofs")


#slide(title: "Identification assumptions")[

]



#slide(title: "Doubly robust identification")[

]

#new-section-slide("Statistical inference")
// example with a simple linear model

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