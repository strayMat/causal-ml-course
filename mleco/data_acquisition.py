from calendar import c
import pandas as pd
from mleco.constants import DIR2DATA


def build_lalonde():
    """
    Build the LaLonde dataset from the [NBER website](https://users.nber.org/~rdehejia/nswdata2.html).
    """
    # RCT population, NSW Data Files (Dehejia-Wahha Sample)
    control = pd.read_csv(
        "http://www.nber.org/~rdehejia/data/nswre74_control.txt", sep="  ", header=None
    )
    control["experimental data"] = 1
    treated = pd.read_csv(
        "http://www.nber.org/~rdehejia/data/nswre74_treated.txt", sep="  ", header=None
    )
    treated["experimental data"] = 1
    # PSID population, NSW Data Files (PSID control, 2490 observations)
    observational = pd.read_csv(
        "http://www.nber.org/~rdehejia/data/psid_controls.txt", sep="  ", header=None
    )
    observational["experimental data"] = 0
    columns = [
        "treatment indicator",  # (1 if treated, 0 if not treated),
        "age",
        "education",
        "Black",  # (1 if black, 0 otherwise)
        "Hispanic",  # (1 if Hispanic, 0 otherwise)
        "married",  # (1 if married, 0 otherwise)
        "nodegree",  # (1 if no degree, 0 otherwise)
        "RE74",  # (earnings in 1974)
        "RE75",  # (earnings in 1975)
        "RE78",  # (earnings in 1978)
        "experimental data",  # (1 if experimental sample, 0 otherwise)
    ]

    lalonde_data = pd.concat([control, treated, observational])
    lalonde_data.columns = columns
    lalonde_data.to_csv(DIR2DATA / "lalonde.csv", index=False)
    return lalonde_data


# def build_ccdrug():
#     ccdrug_data = pd.read_stata(
#         "https://github.com/NickCH-K/causaldata/blob/main/Python/causaldata/ccdrug/ccdrug.dta"
#     )
