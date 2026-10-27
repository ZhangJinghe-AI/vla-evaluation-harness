"""SimplerEnv keeps one SapienRenderer for the process."""

from vla_eval.benchmarks.simpler.benchmark import _cached_renderer_factory


def test_simpler_renderer_factory_reuses_the_first_instance():
    created = []

    def original():
        renderer = object()
        created.append(renderer)
        return renderer

    holder = {"renderer": None}
    factory = _cached_renderer_factory(holder, original)
    first = factory()
    second = factory()
    assert first is second
    assert created == [first]
