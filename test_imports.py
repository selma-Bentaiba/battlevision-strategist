#!/usr/bin/env python3
"""Quick test to verify all imports work"""

print("🎯 Testing BattleVision Strategist imports...")

try:
    from utils.cv_functions import detect_objects, apply_patch, defend_image
    print("✅ CV functions imported successfully")
except Exception as e:
    print(f"❌ CV functions failed: {e}")
    exit(1)

try:
    from utils.game_theory import generate_payoff_matrix, calculate_nash_equilibrium
    print("✅ Game theory functions imported successfully")
except Exception as e:
    print(f"❌ Game theory functions failed: {e}")
    exit(1)

try:
    from utils.report_generator import generate_pdf_report
    print("✅ Report generator imported successfully")
except Exception as e:
    print(f"❌ Report generator failed: {e}")
    exit(1)

print("\n✅ All imports successful! Ready to deploy.")
