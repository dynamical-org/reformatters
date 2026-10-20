from reformatters.google.weathernext_terms import (
    COPYRIGHT,
    LICENSE,
    REAL_TIME_ATTRIBUTION,
    TERMS_SHA256,
    TERMS_URL,
)


def test_terms_constants_are_pinned() -> None:
    assert (
        TERMS_URL
        == "https://storage.googleapis.com/weathernext-public/terms-of-use.pdf"
    )
    assert (
        TERMS_SHA256
        == "5fe155ace6bb11737f15bc735913c5739d31d27374269a8cbfa5f338c4d40654"
    )
    assert COPYRIGHT == "Copyright 2024-6 Google LLC"
    assert LICENSE == (
        "Real-Time Data & Forecasts: GDM Real-Time Weather Forecasting Experimental Data Terms of Use "
        "(https://storage.googleapis.com/weathernext-public/terms-of-use.pdf). "
        "Historical Data: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)."
    )
    assert (
        REAL_TIME_ATTRIBUTION
        == "© 2024-6 Google LLC, whose machine learning models were used to create the experimental data made available under the following licence terms https://storage.googleapis.com/weathernext-public/terms-of-use.pdf. This data is intended for experimental modelling only and is not intended, validated, or approved for real world use."
    )
