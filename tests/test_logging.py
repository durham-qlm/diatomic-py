import logging

import diatomic


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _run_with_log_capture(level, func):
    handler = ListHandler()
    old_handlers = diatomic.logger.handlers[:]
    old_level = diatomic.logger.level
    old_propagate = diatomic.logger.propagate

    diatomic.logger.handlers = [handler]
    diatomic.logger.setLevel(level)
    diatomic.logger.propagate = False
    caught_exception = None
    try:
        func()
    except Exception as exc:
        caught_exception = exc
    finally:
        diatomic.logger.handlers = old_handlers
        diatomic.logger.setLevel(old_level)
        diatomic.logger.propagate = old_propagate

    return [record.getMessage() for record in handler.records], caught_exception


def _messages_at_level(level, func):
    messages, caught_exception = _run_with_log_capture(level, func)
    if caught_exception is not None:
        raise caught_exception
    return messages


def test_log_time_keeps_nested_timings_at_debug_level():
    @diatomic.log_time
    def inner():
        return None

    @diatomic.log_time
    def outer():
        inner()

    messages = _messages_at_level(logging.INFO, outer)

    assert messages[0] == "Starting outer..."
    assert messages[-1].startswith("Finished outer, took: ")
    assert not any("inner" in message for message in messages)


def test_log_time_can_show_nested_timings_at_debug_level():
    @diatomic.log_time
    def inner():
        return None

    @diatomic.log_time
    def outer():
        inner()

    messages = _messages_at_level(logging.DEBUG, outer)

    assert "Starting inner..." in messages
    assert any(message.startswith("Finished inner, took: ") for message in messages)


def test_log_time_reports_failures():
    @diatomic.log_time
    def failing():
        raise RuntimeError("boom")

    messages, caught_exception = _run_with_log_capture(logging.INFO, failing)

    assert isinstance(caught_exception, RuntimeError)
    assert messages[0] == "Starting failing..."
    assert messages[-1].startswith("Failed failing after ")
