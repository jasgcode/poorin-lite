import threading

from repro.accelerator import exclusive_accelerator


def test_a_second_user_waits_for_the_accelerator():
    order: list[str] = []
    first_holds = threading.Event()
    release_first = threading.Event()

    def first() -> None:
        with exclusive_accelerator("first"):
            first_holds.set()
            release_first.wait(timeout=5)
            order.append("first done")

    def second() -> None:
        with exclusive_accelerator("second"):
            order.append("second ran")

    one = threading.Thread(target=first)
    one.start()
    first_holds.wait(timeout=5)
    two = threading.Thread(target=second)
    two.start()
    two.join(timeout=0.3)
    assert two.is_alive(), "the second user must wait while the first holds the accelerator"
    release_first.set()
    one.join(timeout=5)
    two.join(timeout=5)
    assert order == ["first done", "second ran"]
