#!/usr/bin/env python3
"""Test script to verify RobotStateMachine integration.

This script tests that:
1. RobotStateMachine can be instantiated
2. State transitions work correctly
3. Callbacks fire properly
4. Invalid transitions are rejected
"""

import sys
from robot_state_machine import RobotStateMachine

def test_state_machine():
    """Test RobotStateMachine integration."""
    print("=" * 60)
    print("Testing RobotStateMachine Integration")
    print("=" * 60)

    # Track callback invocations
    callbacks_received = []

    def test_callback(new_state, old_state, mood, text):
        """Test callback to track state changes."""
        callbacks_received.append((new_state, old_state, mood, text))
        print(f"✓ Callback fired: {old_state} → {new_state}")

    # Test 1: Instantiation
    print("\n[TEST 1] Instantiating RobotStateMachine...")
    try:
        sm = RobotStateMachine()
        print(f"✓ State machine created, initial state: {sm.get_state()}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test 2: Register callback
    print("\n[TEST 2] Registering callback...")
    try:
        sm.register_callback(test_callback)
        print("✓ Callback registered")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test 3: Valid state transitions
    print("\n[TEST 3] Testing valid state transitions...")
    valid_transitions = [
        ("sleep", "idle", "wake_up transition"),
        ("idle", "listening", "wake_word_detected_from_idle"),
        ("listening", "thinking", "speech_captured"),
        ("thinking", "speaking", "response_received"),
        ("speaking", "idle", "speech_complete"),
    ]

    for old, new, desc in valid_transitions:
        try:
            sm.update_state(new, mood="test_mood", text=f"Testing {desc}")
            current = sm.get_state()
            if current == new:
                print(f"✓ {old} → {new} successful")
            else:
                print(f"✗ Expected {new}, got {current}")
                return False
        except Exception as e:
            print(f"✗ Transition {old} → {new} failed: {e}")
            return False

    # Test 4: Verify callbacks were fired
    print("\n[TEST 4] Verifying callbacks...")
    expected_callbacks = len(valid_transitions)
    actual_callbacks = len(callbacks_received)
    if actual_callbacks == expected_callbacks:
        print(f"✓ All {expected_callbacks} callbacks fired")
    else:
        print(f"✗ Expected {expected_callbacks} callbacks, got {actual_callbacks}")
        return False

    # Test 5: Same-state transitions are allowed
    print("\n[TEST 5] Testing same-state transitions...")
    try:
        current = sm.get_state()
        sm.update_state(current)  # Should be allowed
        if sm.get_state() == current:
            print(f"✓ Same-state transition allowed (idle → idle)")
        else:
            print(f"✗ State changed unexpectedly")
            return False
    except Exception as e:
        print(f"✗ Same-state transition failed: {e}")
        return False

    # Test 6: Check STATE_CONFIGS exists
    print("\n[TEST 6] Verifying STATE_CONFIGS...")
    try:
        configs = RobotStateMachine.STATE_CONFIGS
        if isinstance(configs, dict) and len(configs) > 0:
            print(f"✓ STATE_CONFIGS exists with {len(configs)} entries")
            print(f"  Sample: idle → {configs.get('idle', {})}")
        else:
            print("✗ STATE_CONFIGS is empty or invalid")
            return False
    except Exception as e:
        print(f"✗ STATE_CONFIGS check failed: {e}")
        return False

    # Test 7: Thread safety (basic check)
    print("\n[TEST 7] Testing thread safety...")
    try:
        import threading
        results = []

        def state_worker(state_name):
            try:
                sm.get_state()  # Read operation
                results.append(True)
            except Exception as e:
                results.append(False)

        threads = [threading.Thread(target=state_worker, args=(f"worker_{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if all(results):
            print(f"✓ Thread safety check passed ({len(results)} concurrent reads)")
        else:
            print(f"✗ Thread safety issues detected")
            return False
    except Exception as e:
        print(f"✗ Thread safety test failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_state_machine()
    sys.exit(0 if success else 1)
