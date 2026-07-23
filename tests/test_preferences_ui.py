"""Focused browser coverage for the persistent preferences popup."""

import json

import pytest

pytestmark = pytest.mark.browser


def _preferences_payload(**viewer):
    return {
        "preferences": {"viewer": viewer},
        "schema": {
            "viewer": {
                "theme": ["dark", "light"],
                "rounded_panes": "boolean",
                "ortho_layout": ["big-left", "horizontal"],
                "dimbar_mode": ["compact", "extended"],
            },
            "window": {
                "terminal": ["native", "browser"],
            },
        },
        "overrides": {},
        "defaults": {
            "viewer": {
                "theme": "dark",
                "rounded_panes": True,
                "ortho_layout": None,
                "dimbar_mode": "compact",
            },
            "window": {"terminal": "native"},
        },
        "server_instance_id": "test-server",
    }


def test_preferences_popup_saves_and_applies_viewer_defaults(
    loaded_viewer, sid_3d
):
    page = loaded_viewer(sid_3d)
    payload = _preferences_payload(
        theme="dark",
        rounded_panes=True,
        ortho_layout="big-left",
        dimbar_mode="compact",
    )
    patches = []

    def handle_preferences(route, request):
        if request.method == "PATCH":
            body = json.loads(request.post_data)
            patches.append(body)
            for section, changes in body["changes"].items():
                payload["preferences"].setdefault(section, {}).update(changes)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(payload),
        )

    page.route("**/preferences/*", handle_preferences)
    assert not page.locator("#preferences-hint").is_visible()
    page.locator("#help-hint").click()
    page.locator("#help-preferences").click()
    page.locator("#preferences-overlay.visible").wait_for()

    page.locator('[data-preference="viewer.ortho_layout"]').select_option("horizontal")
    page.wait_for_function("() => orthoLayoutMode === 'horizontal'")
    page.locator('[data-preference="viewer.dimbar_mode"]').select_option("extended")
    page.wait_for_function(
        "() => document.getElementById('info').classList.contains('dimbar-expanded')"
    )
    page.locator('[data-preference="viewer.rounded_panes"]').select_option("false")
    page.wait_for_function("() => !document.body.classList.contains('rounded-panes')")

    assert patches == [
        {
            "changes": {"viewer": {"ortho_layout": "horizontal"}},
            "server_instance_id": "test-server",
        },
        {
            "changes": {"viewer": {"dimbar_mode": "extended"}},
            "server_instance_id": "test-server",
        },
        {
            "changes": {"viewer": {"rounded_panes": False}},
            "server_instance_id": "test-server",
        },
    ]
    assert page.locator("#preferences-status").text_content() == "Saved"


def test_preferences_popup_dismisses_and_labels_launch_settings(
    loaded_viewer, sid_3d
):
    page = loaded_viewer(sid_3d)
    payload = _preferences_payload()
    page.route(
        "**/preferences/*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(payload),
        ),
    )

    page.locator("#help-hint").click()
    page.locator("#help-preferences").click()
    page.locator("#preferences-overlay.visible").wait_for()
    row = page.locator('[data-preference="window.terminal"]').locator("xpath=..")
    assert "next terminal launch" in row.text_content().lower()
    assert page.locator("#preferences-overlay").get_by_text("Built-in default").count() == 0
    theme_options = page.locator(
        '[data-preference="viewer.theme"] option'
    ).all_text_contents()
    assert theme_options == ["Dark", "Light"]
    assert page.locator('[data-preference="viewer.theme"]').input_value() == "dark"
    assert page.locator(
        '[data-preference="viewer.rounded_panes"]'
    ).input_value() == "true"
    assert page.locator(
        '[data-preference="viewer.dimbar_mode"]'
    ).input_value() == "compact"
    launch_options = page.locator(
        '[data-preference="window.terminal"] option'
    ).all_text_contents()
    assert launch_options == ["Native window", "System browser"]
    assert page.locator('[data-preference="window.terminal"]').input_value() == "native"

    page.keyboard.press("Escape")
    page.wait_for_function(
        "() => !document.getElementById('preferences-overlay').classList.contains('visible')"
    )


def test_preferences_initialize_ortho_and_dimbar_defaults(
    loaded_viewer, sid_3d, tmp_path, monkeypatch
):
    import arrayview._config as config

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[viewer]\northo_layout = "big-left"\ndimbar_mode = "extended"\n'
    )
    monkeypatch.setattr(config, "CONFIG_PATH", str(config_path))

    page = loaded_viewer(sid_3d)

    assert page.evaluate("() => orthoLayoutMode") == "big-left"
    page.wait_for_function(
        "() => document.getElementById('info').classList.contains('dimbar-expanded')"
    )

    page.locator("#help-hint").click()
    page.locator("#help-preferences").click()
    page.locator("#preferences-overlay.visible").wait_for()
    page.locator("#preferences-overlay").click(position={"x": 2, "y": 2})
    page.wait_for_function(
        "() => !document.getElementById('preferences-overlay').classList.contains('visible')"
    )
