# Copyright 2025 Prism Shadow. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from contextlib import suppress
from typing import Any, Awaitable, TypeVar


T = TypeVar("T")


class AbortSignal:
    """Abort signal that can also trigger its own aborted state."""

    def __init__(self) -> None:
        self._aborted = False
        self._reason: Any = None
        self._waiters: set[asyncio.Future[None]] = set()

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def reason(self) -> Any:
        return self._reason

    def abort(self, reason: Any = None) -> None:
        if self._aborted:
            return

        self._aborted = True
        self._reason = reason

        waiters = tuple(self._waiters)
        self._waiters.clear()
        for waiter in waiters:
            loop = waiter.get_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(_set_waiter_result, waiter)
            else:
                _set_waiter_result(waiter)

    async def wait(self) -> None:
        if self._aborted:
            return

        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        self._waiters.add(waiter)
        if self._aborted:
            self._waiters.discard(waiter)
            _set_waiter_result(waiter)
        try:
            await waiter
        finally:
            self._waiters.discard(waiter)

    def throw_if_aborted(self) -> None:
        if self._aborted:
            raise _cancelled_error(self._reason)


async def run_with_abort(awaitable: Awaitable[T], signal: AbortSignal) -> T:
    """Run an awaitable and cancel it when the signal is aborted."""

    task = asyncio.ensure_future(awaitable)

    if signal.aborted:
        task.cancel(signal.reason)
        with suppress(asyncio.CancelledError):
            await task
        raise _cancelled_error(signal.reason)

    abort_task = asyncio.create_task(signal.wait())

    try:
        done, _ = await asyncio.wait((task, abort_task), return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return await task

        task.cancel(signal.reason)
        with suppress(asyncio.CancelledError):
            await task
        raise _cancelled_error(signal.reason)
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        raise
    finally:
        if not abort_task.done():
            abort_task.cancel()
            with suppress(asyncio.CancelledError):
                await abort_task


def _set_waiter_result(waiter: asyncio.Future[None]) -> None:
    if not waiter.done():
        waiter.set_result(None)


def _cancelled_error(reason: Any) -> asyncio.CancelledError:
    if reason is None:
        return asyncio.CancelledError()

    return asyncio.CancelledError(reason)
