from reformatters.common.pydantic import FrozenBaseModel


class WxOpticonTrigger(FrozenBaseModel):
    """A wxopticon source-arrival event a dataset wants its update triggered by.

    `product_id` is a wxopticon product (e.g. "external-noaa-gfs-aws"). `trigger`
    is one of "started", "complete", or "progress:<lead_group>" (e.g. "progress:f384").
    See docs/webhooks.md.
    """

    product_id: str
    trigger: str

    @property
    def is_complete(self) -> bool:
        return self.trigger == "complete"
