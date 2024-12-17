// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.3.1": *
#import "@preview/embiggen:0.0.1": * // LaTeX-like delimiter sizing for Typst

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

#title-slide(
  author: [Authors],
  title: "Machine Learning for econometrics",
  subtitle: "Synthetic Controls",
  date: "January 10, 2025",
  //extra: "Extra"
)

#slide(title: "Table of contents")[
  #metropolis-outline
]

#new-section-slide("Motivation")


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
  == Resources
  
- https://web.stanford.edu/~swager/stats361.pdf
- https://www.mixtapesessions.io/
- https://alejandroschuler.github.io/mci/
]


#slide[
  == Bibliography
  
#bibliography("biblio.bib")  
]
