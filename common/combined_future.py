import concurrent
from concurrent.futures import Future
from typing import Any

"""
This whole file appreciatively copied from https://gist.github.com/Klotzi111/9ab06b0380702cd5f4044c7529bdc096
"""


class CombinedFuture[T](Future[Future[T] | None]):
    """
    This class provides "waiting" mechanisms similar to concurrent.futures.wait(...) except that there is no blocking wait.
    This class extends concurrent.futures.Future and thus it can be used like any other Future.
    You can use the .result() and .done() (and other) methods and also use this class with the aforementioned concurrent.futures.wait function.

    This class is especially useful when you want to combine multiple futures with and (&&) and or (||) logic.
    Example:
    Consider you have multiple parallel tasks (as futures) and a future that will be completed once your function should return (canellation_future).
    You want to wait until all tasks finish normally or the canellation_future is completed.
    With the standard python library this is not possible because concurrent.futures.wait(...) can either wait for all futures or one.
    Using ALL_COMPLETED will never work. And FIRST_COMPLETED would return also if only one task_futures was completed.
    The following code uses CombinedFuture to solve this problem.

    .. code-block:: python

            def create_task() -> Future:
                    # TODO add logic that completes this future
                    return Future()

            # can be completed any time
            cancellation_future = Future()
            task_futures = [create_task(), create_task()]

            task_combined_future = CombinedFuture(*task_futures, complete_when=concurrent.futures.ALL_COMPLETED)
            done, not_done = concurrent.futures.wait([cancellation_future, task_combined_future], timeout=None, return_when=concurrent.futures.ALL_COMPLETED)

            if cancellation_future in done:
                    print("cancellation_future was completed")
            else:
                    print("task_combined_future was completed")
    """

    def __init__(
        self,
        *futures: Future[T],
        complete_when: str = concurrent.futures.ALL_COMPLETED,
    ) -> None:
        self.complete_when = complete_when
        self.futures = set(futures)
        self.completed_futures: set[Future[T]] = set()

        super().__init__()

        for future in self.futures:
            future.add_done_callback(self._future_completed_callback)

    def _set_result_safe(self, result: Any) -> None:
        try:
            self.set_result(result)
        except Exception:  # noqa: S110
            # this might happen when the future had its result already set
            # this can happen when:
            # a second future completes or multiple at "the same time"
            # or the user called set_result or changed the complete_when attribute. both is not supported
            pass

    def _future_completed_callback(self, future: Future[T]) -> None:
        self.completed_futures.add(future)

        if self.complete_when == concurrent.futures.FIRST_COMPLETED:
            # no count check required because we only need one and we just added our future
            self._set_result_safe(future)
            return
        elif self.complete_when == concurrent.futures.FIRST_EXCEPTION:
            if future.exception(timeout=0) is not None:
                # future completed with exception
                self._set_result_safe(future)
        # else: should be concurrent.futures.ALL_COMPLETED
        # but we also want this logic in the FIRST_EXCEPTION case
        if self.completed_futures == self.futures:
            self._set_result_safe(None)
