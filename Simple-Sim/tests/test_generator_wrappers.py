from simple_sim.generator_2d import normalize_image_filters as normalize_from_wrapper
from simple_sim.generators.filter_settings import normalize_image_filters


def test_generator_2d_wrapper_kept() -> None:
    assert normalize_from_wrapper({"enable": False})["enable"] is False
    assert normalize_image_filters({"enable": False})["enable"] is False
