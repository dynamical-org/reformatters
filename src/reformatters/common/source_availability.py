from collections.abc import Sequence

import httpx
from pydantic import BaseModel

from reformatters.common.pydantic import FrozenBaseModel
from reformatters.common.types import Timestamp

SUMMARY_URL = "https://assets.dynamical.org/wxopticon/summary.json"


class Facet(BaseModel):
    name: str
    status: str


class LeadGroup(BaseModel):
    max_lead: int
    status: str
    facets: list[Facet] = []


class ForecastRun(BaseModel):
    init_time: Timestamp
    lead_groups: list[LeadGroup] = []


class Product(BaseModel):
    id: str
    recent_inits: list[ForecastRun]


class Summary(BaseModel):
    products: list[Product]

    @classmethod
    def fetch(cls) -> Summary:
        response = httpx.get(SUMMARY_URL, timeout=30)
        response.raise_for_status()
        return cls.model_validate_json(response.content)


class SourceAvailability(FrozenBaseModel):
    product_id: str
    lead_hours: int
    components: tuple[str, ...] = ()

    def available_inits(self, summary: Summary) -> set[Timestamp]:
        products = [p for p in summary.products if p.id == self.product_id]
        if len(products) != 1:
            raise ValueError(
                f"wxopticon summary must contain exactly one product {self.product_id!r}"
            )
        product = products[0]
        return {
            run.init_time.tz_localize(None)
            for run in product.recent_inits
            if self._complete(run.lead_groups)
        }

    def _complete(self, groups: Sequence[LeadGroup]) -> bool:
        relevant = [g for g in groups if g.max_lead <= self.lead_hours]
        if not any(g.max_lead == self.lead_hours for g in relevant):
            return False
        if not self.components:
            return all(g.status == "complete" for g in relevant)
        return all(
            any(
                facet.name == f"component:{component}" and facet.status == "complete"
                for facet in group.facets
            )
            for group in relevant
            for component in self.components
        )
