import hashlib
from collections.abc import Mapping
from pathlib import Path

from reformatters.common.config_models import DatasetAttributes

TERMS_URL = "https://storage.googleapis.com/weathernext-public/terms-of-use.pdf"
TERMS_SHA256 = "5fe155ace6bb11737f15bc735913c5739d31d27374269a8cbfa5f338c4d40654"
LICENSE = (
    "Real-Time Data & Forecasts: GDM Real-Time Weather Forecasting Experimental "
    f"Data Terms of Use ({TERMS_URL}). Historical Data: CC BY 4.0 "
    "(https://creativecommons.org/licenses/by/4.0/)."
)
COPYRIGHT = "Copyright 2024-6 Google LLC"
LEGALLY_BINDING_NOTICE = "By using this information, you agree to the Terms of Use found at https://storage.googleapis.com/weathernext-public/terms-of-use.pdf"
REAL_TIME_ATTRIBUTION = "© 2024-6 Google LLC, whose machine learning models were used to create the experimental data made available under the following licence terms https://storage.googleapis.com/weathernext-public/terms-of-use.pdf. This data is intended for experimental modelling only and is not intended, validated, or approved for real world use."
USAGE_RESTRICTIONS = "The GDM Real-Time Weather Forecasting Experimental Data Terms of Use (Section 1(c), and Section 4: Use Restrictions (b)) do not permit making real-time data available in a Restricted Country. Section 1(c) lists them as “Japan, South Korea, Indonesia, Cuba, Iran, North Korea, Crimea and the so-called Donetsk People’s Republic and Luhansk People’s Republic”. The data service at wn.dynamical.org refuses all WeatherNext data in those locations, whatever its age, and also refuses requests whose location cannot be determined, including requests over Tor."  # noqa: RUF001
NON_ENDORSEMENT = "dynamical.org and this dataset are not sponsored, endorsed, or granted official status by Google or any meteorological agency or department. Unless otherwise stated, the Real-Time Experimental Data (i) is not based on data published by; (ii) has not been produced in collaboration with; and (iii) has not been endorsed by any government meteorological agency or department and in no way replaces official alerts, warnings or notices published by such agencies or departments."


def render_notice_files(
    attributes: DatasetAttributes,
    *,
    product: str,
    modifications: str,
    usage_restrictions: str,
    non_endorsement: str,
) -> dict[str, bytes]:
    paragraphs = (
        COPYRIGHT,
        f"{attributes.name} ({attributes.dataset_id}) is published by dynamical.org from {product} experimental forecast data.",
        "Data relating to a time less than 1 hour ago and the future is Real-Time Experimental Data governed by the GDM Real-Time Weather Forecasting Experimental Data Terms of Use. TERMS_OF_USE.pdf beside this file is a copy of those terms, and LEGALLY_BINDING_TERMS_OF_USE.txt carries Google's notice. Data relating to a time 1 hour ago or more is licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). A forecast value relates to the time in its valid_time coordinate (init_time + lead_time).",
        LEGALLY_BINDING_NOTICE,
        f"Modifications: {modifications}",
        f"Attribution: {attributes.attribution}",
        usage_restrictions,
        non_endorsement,
    )
    return {
        "TERMS_OF_USE.pdf": Path(__file__)
        .with_name("weathernext_terms_of_use.pdf")
        .read_bytes(),
        "LEGALLY_BINDING_TERMS_OF_USE.txt": (LEGALLY_BINDING_NOTICE + "\n").encode(),
        "NOTICE.txt": ("\n\n".join(paragraphs) + "\n").encode(),
    }


def notice_revision(files: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, data in sorted(files.items()):
        digest.update(name.encode() + b"\0" + data + b"\0")
    return digest.hexdigest()[:12]


def repo_notice_files(
    attributes: DatasetAttributes,
    *,
    product: str,
    modifications: str,
    usage_restrictions: str,
    non_endorsement: str,
) -> dict[str, bytes]:
    files = render_notice_files(
        attributes,
        product=product,
        modifications=modifications,
        usage_restrictions=usage_restrictions,
        non_endorsement=non_endorsement,
    )
    revision = notice_revision(files)
    return {f"notices/{revision}/{name}": data for name, data in files.items()}
