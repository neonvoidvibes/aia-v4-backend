#!/usr/bin/env python3
"""
Test script to verify cross-group read logic.

This script simulates the event filtering logic to ensure personal events
are excluded and only group events are included when allow_cross_group_read is enabled.
"""


def test_cross_group_read_logic():
    """Test the cross-group read event filtering logic."""

    # Simulate event profile data
    test_cases = [
        {
            "name": "Cross-group read enabled with group events",
            "event_id": "0000",
            "allow_cross_group_read": True,
            "allowed_group_events": {"event-a", "event-b", "event-c"},
            "allowed_personal_events": {"personal-1"},
            "expected_tier3": {"event-a", "event-b", "event-c"}
        },
        {
            "name": "Cross-group read disabled",
            "event_id": "0000",
            "allow_cross_group_read": False,
            "allowed_group_events": {"event-a", "event-b"},
            "allowed_personal_events": {"personal-1"},
            "expected_tier3": set()  # Empty because we're in 0000 and it's excluded
        },
        {
            "name": "Cross-group read enabled but no group events",
            "event_id": "0000",
            "allow_cross_group_read": True,
            "allowed_group_events": set(),
            "allowed_personal_events": {"personal-1"},
            "expected_tier3": set()
        },
        {
            "name": "In non-0000 event (standard tier3 logic)",
            "event_id": "event-a",
            "allow_cross_group_read": True,  # Irrelevant when not in 0000
            "allowed_events": {"0000", "event-a", "event-b", "event-c"},
            "allowed_group_events": {"event-a", "event-b", "event-c"},
            "allowed_personal_events": set(),
            "expected_tier3": {"event-b", "event-c"}  # Excludes current and 0000
        },
        {
            "name": "Cross-group with mixed personal and group events",
            "event_id": "0000",
            "allow_cross_group_read": True,
            "allowed_group_events": {"event-a", "event-b"},
            "allowed_personal_events": {"personal-1", "personal-2"},
            "expected_tier3": {"event-a", "event-b"}  # Only group, no personal
        }
    ]

    print("Testing cross-group read logic\n" + "="*60)

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"  Event ID: {test['event_id']}")
        print(f"  Allow cross-group read: {test['allow_cross_group_read']}")
        print(f"  Group events: {test['allowed_group_events']}")
        print(f"  Personal events: {test['allowed_personal_events']}")

        # Simulate the logic from api_server.py
        event_id = test['event_id']
        allow_cross_group_read = test['allow_cross_group_read']
        allowed_group_events = test['allowed_group_events']
        allowed_events = test.get('allowed_events', set())

        tier3_allow_events = set()

        if event_id == '0000' and allow_cross_group_read and allowed_group_events:
            # Cross-group read enabled: include all accessible group events (excluding personal)
            tier3_allow_events = set(allowed_group_events)
            print(f"  → Cross-group read branch activated")
        else:
            # Standard tier3 logic: excludes current event and shared namespace
            if allowed_events:
                tier3_allow_events = {ev for ev in allowed_events if ev not in {event_id, '0000'}}
            if allowed_group_events:
                tier3_allow_events = {ev for ev in tier3_allow_events if ev in allowed_group_events}
            print(f"  → Standard tier3 branch activated")

        print(f"  Result tier3 events: {tier3_allow_events}")
        print(f"  Expected: {test['expected_tier3']}")

        # Verify personal events are never included
        personal_leaked = tier3_allow_events & test['allowed_personal_events']
        if personal_leaked:
            print(f"  ❌ FAIL: Personal events leaked into tier3: {personal_leaked}")
        elif tier3_allow_events == test['expected_tier3']:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL: Mismatch")
            print(f"     Diff: expected-actual={test['expected_tier3'] - tier3_allow_events}")
            print(f"           actual-expected={tier3_allow_events - test['expected_tier3']}")

    print("\n" + "="*60)
    print("Test complete!")


if __name__ == "__main__":
    test_cross_group_read_logic()
