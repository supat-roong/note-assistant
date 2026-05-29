from note_assistant.errors import ErrorBus, error_bus


def test_emit_calls_subscriber():
    received = []
    bus = ErrorBus()
    bus.subscribe(lambda s, m, sev: received.append((s, m, sev)))
    bus.emit("transcriber", "something broke")
    assert received == [("transcriber", "something broke", "error")]


def test_emit_with_warning_severity():
    received = []
    bus = ErrorBus()
    bus.subscribe(lambda s, m, sev: received.append(sev))
    bus.emit("summarizer", "queue full", "warning")
    assert received == ["warning"]


def test_multiple_subscribers():
    a, b = [], []
    bus = ErrorBus()
    bus.subscribe(lambda s, m, sev: a.append(m))
    bus.subscribe(lambda s, m, sev: b.append(m))
    bus.emit("audio", "device lost")
    assert a == ["device lost"]
    assert b == ["device lost"]


def test_no_subscribers_does_not_raise():
    bus = ErrorBus()
    bus.emit("test", "no one listening")  # should not raise


def test_global_error_bus_is_singleton():
    from note_assistant.errors import error_bus as bus2
    assert error_bus is bus2
