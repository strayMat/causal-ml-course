// Get Polylux from the official package repository

// documentation link : https://typst.app/universe/package/polylux

#import "@preview/polylux:0.3.1": *
#import "@preview/embiggen:0.0.1": * // LaTeX-like delimiter sizing for Typst
#import "@preview/showybox:2.0.1": showybox

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

#let eq(body, size: 1.1em) = {
  set align(center)
  set text(size: size)
  body
}

// assumption box
//
//
#let my_box(title, color, body) = {
  showybox(
    title-style: (
      color: black,
      weight: "light",
      boxed-style: (
        anchor: (
          x: left,
          y: horizon,
        ),
        radius: 10pt,
      ),
    ),
    frame: (
      border-color: color.darken(50%),
      title-color: color.lighten(60%),
      body-color: color.lighten(80%),
    ),
    title: title,
    align(center, body),
  )
}

#let hyp_box(title: "Assumption", body) = {
  my_box(title, rgb("#d39e57"), body)
}


#let def_box(title: "Definition", body) = {
  my_box(title, rgb("#57b4d3"), body)
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
    show: pad.with(
      left: 1em,
      top: 0em,
      right: 1em,
      bottom: 0em,
    ) // super important to have a proper padding (not 1/3 of the slide blank...)
    set text(fill: m-dark-teal)
    body
  }

  logic.polylux-slide(content)
}

#title-slide(
  author: [Authors],
  title: "Machine Learning for econometrics",
  subtitle: "Event studies: Causal methods for pannel data",
  date: "February, 11th, 2025",
)

#slide(title: "Table of contents")[
  #metropolis-outline
]

#new-section-slide("Motivation")


#new-section-slide("A visual reminder on difference-in-differences")

#slide(title: "Difference-in-differences")[

  == Difference-in-differences

  - Introduced by
  - A method to estimate the effect of a treatment on a treated unit
  - The treatment effect is estimated by comparing the evolution of the treated unit to a control unit
  - The difference between the two differences is the treatment effect
]

#slide(title: "Target effect: Sample Average Treatment effect on the Treated, SATT")[
  #eq($tau_(text("SATT")) = 1 / (|{i \: D_i=1}|) sum_(i:D_i=1) 1 / (T-H) sum_(t=H+1)^T Y_(i t)(1) - Y_(i t)(0)$)
]

#slide(title: "DID estimator")[
  #eq($hat(tau_(text("DID"))) = 1 / (|{i \: D_i=1}|) sum_(i:D_i=1) [1 / (T-H) sum_(t=H+1)^T Y_(i t) - 1 / H sum_(t=1)^H Y_(i t)] - 1 / (|{i \: D_i=0}|) sum_(i:D_i=0) [1 / (T-H) sum_(t=H+1)^T Y_(i t) - 1 / H sum_(t=1)^H Y_(i t)]$)
]

#slide(title: "Temporal consistency assumption")[
  == No anticipation of the treatment (in practice, not always true)
  #hyp_box[
    #eq($Y_(i t)(0) = Y_(i t)(1) forall t = 1,..., H.$)
  ]
]

#slide(title: "Parallel trend assumption")[
  == Main and #alert[strong] assumption of the DID method

  #hyp_box[
    #eq($EE [Y_(i t)(0, infinity) - Y_(i 1)(0, infinity)] = beta_t, t = 2,..., T.$)
  ]
]


#slide(title: "Parallel trend assumption")[
  Under the temporal consistency and the parallel trend assumptions, the DID estimator is unbiased ie. $EE[hat(tau_(text("DID")) - tau_(text("SATT")))]=0$

  See @wager2024causal for a clear proof.
]

#slide(title: "Estimation: link with two way fixed effect")[

  In practice, DID is estimated with a two-way fixed effect model (TWFE):

  #eq[$Y_(i t) tilde alpha_i + beta_t + A_(i t) tau$ where $A_(i t) = D_i * (t<=H)$]

  - $alpha_i$ capture the individual fixed effect
  - $beta_t$ capture the time fixed effect (under pararell trend)

  This link can be seen with the parallel trend assumption: $beta_t= $
]

#new-section-slide("Interrupted Time Series")

#new-section-slide("Synthetic Controls")

#slide()[
  == Synthetic Controls
  Introduced in @abadie2003economic and @abadie2010synthetic, well described in @abadie2021using
  - A method for estimating the effect of a treatment on a single unit
  - The treatment unit is compared to a weighted average of control units
  - The weights are chosen to minimize the difference between the treated unit and the synthetic control

  Example for the effect of taxes on sugar-based product consumption in @puig2021impact, review of usage in healthcare @bouttell2018synthetic.
]

#new-section-slide("Time-series modelisation: other methods")

#slide(title: "State space models")[

]


#slide(title: "Take-away")[

]

#new-section-slide("Python hands-on")

#slide(title: [To your notebooks 🧑‍💻!])[
  - url: https://github.com/strayMat/causal-ml-course/tree/main/notebooks
]


#let bibliography = bibliography("biblio.bib", style: "apa")

#set align(left)
#set text(size: 20pt, style: "italic")

#slide(title: "Bibliography")[
  #bibliography
]
