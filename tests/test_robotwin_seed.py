"""RoboTwin episode seeds stay fixed across reruns."""

from vla_eval.benchmarks.robotwin.benchmark import (
    _cached_renderer_factory,
    _is_renderer_crash,
    _release_robotwin_env,
    robotwin_episode_seed,
)


def test_episode_lanes_do_not_overlap():
    task = {"seed_base": 100000, "episode_idx": 0}
    assert robotwin_episode_seed(task, 0) == 100000
    assert robotwin_episode_seed(task, 1) == 100001
    task["episode_idx"] = 1
    assert robotwin_episode_seed(task, 0) == 100100
    assert robotwin_episode_seed({"seed": 100004}, 2) == 100006


def test_renderer_factory_reuses_the_first_instance():
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
    assert holder["renderer"] is first


class _Scene:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


class _Env:
    def __init__(self):
        self.scene = _Scene()
        self.renderer = object()
        self.engine = object()
        self.closed = False

    def close_env(self, clear_cache=False):
        self.closed = True


def test_release_keeps_renderer_and_clears_scene():
    env = _Env()
    holder = {"renderer": None}
    scene = env.scene
    renderer = env.renderer
    _release_robotwin_env(env, renderer_holder=holder)
    assert scene.cleared
    assert env.closed
    assert env.scene is None
    assert env.renderer is None
    assert holder["renderer"] is renderer


def test_release_drops_renderer_only_when_asked():
    env = _Env()
    holder = {"renderer": env.renderer}
    _release_robotwin_env(env, renderer_holder=holder, drop_renderer=True)
    assert holder["renderer"] is None


def test_camera_buffer_failure_counts_as_renderer_crash():
    assert _is_renderer_crash(RuntimeError("cannot create buffer"))
    assert not _is_renderer_crash(RuntimeError("object not found"))
