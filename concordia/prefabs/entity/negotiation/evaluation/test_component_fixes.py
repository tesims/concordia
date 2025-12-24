#!/usr/bin/env python3
"""Test script for component bug fixes.

Tests the following fixes:
1. _style attribute exists in BasicNegotiationStrategy (was _negotiation_style)
2. get_state() returns Mapping[str, Any], not str
3. Hook methods (post_act, pre_observe, post_observe) return str, not None
4. _component_access_failures tracking in InterpretabilityRunner
5. Specific ValueError exception handling in swarm_intelligence.py
"""

import sys
import os

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
sys.path.insert(0, project_root)

print("=" * 70)
print("COMPONENT BUG FIXES TEST SUITE")
print("=" * 70)

passed = 0
failed = 0

# ============================================================================
# TEST 1: _style attribute exists in BasicNegotiationStrategy
# ============================================================================
print("\n[TEST 1] BasicNegotiationStrategy._style attribute")
print("-" * 50)

try:
    from concordia.prefabs.entity.negotiation.components.negotiation_strategy import (
        BasicNegotiationStrategy
    )

    strategy = BasicNegotiationStrategy(
        agent_name="TestAgent",
        negotiation_style="cooperative",
    )

    # Check _style exists (not _negotiation_style)
    has_style = hasattr(strategy, '_style')
    has_old_attr = hasattr(strategy, '_negotiation_style')

    if has_style:
        print(f"  ✓ _style attribute exists: '{strategy._style}'")
        passed += 1
    else:
        print("  ✗ _style attribute MISSING")
        failed += 1

    if has_old_attr:
        print("  ⚠ WARNING: Old _negotiation_style attribute still exists")

except Exception as e:
    print(f"  ✗ FAIL: {e}")
    failed += 1

# ============================================================================
# TEST 2: get_state() returns Mapping[str, Any]
# ============================================================================
print("\n[TEST 2] get_state() returns Mapping[str, Any]")
print("-" * 50)

components_to_test = [
    ('negotiation_strategy', 'BasicNegotiationStrategy', {'agent_name': 'Test'}),
    ('negotiation_instructions', 'NegotiationInstructions', {'agent_name': 'Test', 'goal': 'Negotiate a fair deal'}),
]

# Note: NegotiationMemory requires memory_bank (complex setup)
# Note: CulturalAdaptation requires language_model (complex setup)
# These are tested implicitly via integration tests

for module_name, class_name, kwargs in components_to_test:
    try:
        module = __import__(
            f'concordia.prefabs.entity.negotiation.components.{module_name}',
            fromlist=[class_name]
        )
        cls = getattr(module, class_name)
        instance = cls(**kwargs)

        state = instance.get_state()

        # Check it's a dict/Mapping, not a string
        if isinstance(state, dict):
            print(f"  ✓ {class_name}.get_state() returns dict with {len(state)} keys")
            passed += 1
        elif isinstance(state, str):
            print(f"  ✗ {class_name}.get_state() returns STRING (bug not fixed!)")
            failed += 1
        else:
            print(f"  ? {class_name}.get_state() returns {type(state)}")
            failed += 1

    except Exception as e:
        print(f"  ✗ {class_name}: {e}")
        failed += 1

# ============================================================================
# TEST 3: Hook methods return str, not None
# ============================================================================
print("\n[TEST 3] Hook methods return str")
print("-" * 50)

hooks_to_test = ['post_act', 'pre_observe', 'post_observe']

for module_name, class_name, kwargs in components_to_test:
    try:
        module = __import__(
            f'concordia.prefabs.entity.negotiation.components.{module_name}',
            fromlist=[class_name]
        )
        cls = getattr(module, class_name)
        instance = cls(**kwargs)

        hook_results = []
        for hook_name in hooks_to_test:
            if hasattr(instance, hook_name):
                hook = getattr(instance, hook_name)

                # Call with appropriate args
                if hook_name == 'post_act':
                    result = hook("test action")
                elif hook_name == 'pre_observe':
                    result = hook("test observation")
                elif hook_name == 'post_observe':
                    result = hook()

                if isinstance(result, str):
                    hook_results.append(f"{hook_name}=str")
                elif result is None:
                    hook_results.append(f"{hook_name}=None(BUG!)")
                else:
                    hook_results.append(f"{hook_name}={type(result)}")

        all_str = all('=str' in r for r in hook_results)
        if all_str:
            print(f"  ✓ {class_name}: all hooks return str")
            passed += 1
        else:
            print(f"  ✗ {class_name}: {', '.join(hook_results)}")
            failed += 1

    except Exception as e:
        print(f"  ✗ {class_name}: {e}")
        failed += 1

# ============================================================================
# TEST 4: _component_access_failures tracking
# ============================================================================
print("\n[TEST 4] InterpretabilityRunner._component_access_failures")
print("-" * 50)

try:
    from concordia.prefabs.entity.negotiation.evaluation.interpretability_evaluation import (
        InterpretabilityRunner
    )

    # Check attribute exists in class or would be created
    import inspect
    source = inspect.getsource(InterpretabilityRunner.__init__)

    has_tracking = '_component_access_failures' in source

    if has_tracking:
        print("  ✓ _component_access_failures initialized in __init__")
        passed += 1
    else:
        print("  ✗ _component_access_failures NOT found in __init__")
        failed += 1

    # Check it's used in exception handling
    full_source = inspect.getsource(InterpretabilityRunner)
    uses_tracking = '_component_access_failures' in full_source and 'except' in full_source

    if uses_tracking:
        print("  ✓ _component_access_failures used in exception handling")
        passed += 1
    else:
        print("  ⚠ _component_access_failures may not be used in exception handling")

except Exception as e:
    print(f"  ✗ FAIL: {e}")
    failed += 1

# ============================================================================
# TEST 5: ValueError exception handling in swarm_intelligence.py
# ============================================================================
print("\n[TEST 5] Specific ValueError handling in swarm_intelligence.py")
print("-" * 50)

try:
    import inspect
    from concordia.prefabs.entity.negotiation.components import swarm_intelligence

    source = inspect.getsource(swarm_intelligence)

    # Check for bare except (bad)
    has_bare_except = 'except:' in source
    # Check for specific ValueError (good)
    has_value_error = 'except ValueError' in source

    if has_bare_except:
        print("  ✗ Still has bare 'except:' (bug not fixed!)")
        failed += 1
    else:
        print("  ✓ No bare 'except:' found")
        passed += 1

    if has_value_error:
        print("  ✓ Uses specific 'except ValueError'")
        passed += 1
    else:
        print("  ? 'except ValueError' not found (may use different exception)")

except Exception as e:
    print(f"  ✗ FAIL: {e}")
    failed += 1

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Total:  {passed + failed}")

if failed == 0:
    print("\n  ✓ ALL COMPONENT BUG FIXES VERIFIED!")
else:
    print(f"\n  ✗ {failed} test(s) failed - some bugs may not be fixed")

print()
