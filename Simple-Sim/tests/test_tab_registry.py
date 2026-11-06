from gui.tabs.core.registry import TabRegistry


def test_tab_registry_contains_expected_first_tab() -> None:
    specs = list(TabRegistry.iter_specs())
    assert specs
    assert specs[0].key == "pipeline"
    assert specs[0].title == "Pipeline Control"


def test_tab_registry_keys_match_specs() -> None:
    specs = list(TabRegistry.iter_specs())
    assert TabRegistry.keys() == [s.key for s in specs]


def test_tab_registry_groups_cover_every_tab() -> None:
    grouped = TabRegistry.grouped()
    flat = [s.key for _g, specs in grouped for s in specs]
    assert sorted(flat) == sorted(TabRegistry.keys())
    for spec in TabRegistry.iter_specs():
        assert spec.description, spec.key
