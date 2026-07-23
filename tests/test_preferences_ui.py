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
                "terminal": ["browser", "inline", "native", "none", "vscode"],
            },
        },
        "overrides": {},
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
    page.locator("#preferences-hint").click()
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

    page.locator("#preferences-hint").click()
    page.locator("#preferences-overlay.visible").wait_for()
    row = page.locator('[data-preference="window.terminal"]').locator("xpath=..")
    assert "next terminal launch" in row.text_content().lower()

    page.keyboard.press("Escape")
    page.wait_for_function(
        "() => !document.getElementById('preferences-overlay').classList.contains('visible')"
    )

    page.locator("#preferences-hint").click()
    page.locator("#preferences-overlay.visible").wait_for()
    page.locator("#preferences-overlay").click(position={"x": 2, "y": 2})
    page.wait_for_function(
        "() => !document.getElementById('preferences-overlay').classList.contains('visible')"
    )
