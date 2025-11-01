#!/usr/bin/env python3
"""
Quick diagnostic script to check if imports are correct
"""

print("Checking imports...")
print("=" * 60)

try:
    print("1. Testing: from se_st_upgrade.models.se_st_combined import SE_ST_CombinedModel")
    from se_st_upgrade.models.se_st_combined import SE_ST_CombinedModel
    print("   ✅ SUCCESS")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

try:
    print("\n2. Testing: from se_st_upgrade.data.perturbation_dataset import PerturbationDataset")
    from se_st_upgrade.data.perturbation_dataset import PerturbationDataset
    print("   ✅ SUCCESS")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print("\n" + "=" * 60)
print("If both tests passed, imports are correct!")
print("If any failed, run: git pull origin main")
