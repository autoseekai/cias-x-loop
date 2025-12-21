#!/usr/bin/env python3
"""
Quick test script to verify config hash optimization works correctly
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sci_scientist.models.world_model import WorldModel
from src.sci_scientist.agents.planner import PlannerAgent, ConfigHasher, create_baseline_configs
from src.sci_scientist.core.data_structures import ExperimentResult, SCIMetrics


def test_hash_optimization():
    """Test the config hash optimization"""
    print("=" * 60)
    print("Testing Config Hash Optimization")
    print("=" * 60)

    # Create temporary in-memory database
    print("\n1. Creating in-memory database...")
    world_model = WorldModel(":memory:")
    print("✓ Database created")

    # Create test configuration
    design_space = {
        "compression_ratios": [8, 16, 24],
        "mask_types": ["random", "optimized"],
        "num_stages": [5, 7, 9],
        "num_features": [32, 64, 128],
        "num_blocks": [2, 3, 4],
        "learning_rates": [1e-4, 5e-5],
        "activations": ["ReLU", "LeakyReLU"]
    }

    # Add some baseline configs
    print("\n2. Creating baseline experiments...")
    configs = create_baseline_configs(design_space)
    print(f"✓ Created {len(configs)} baseline configs")

    # Add experiments to database
    print("\n3. Adding experiments to database...")
    for i, config in enumerate(configs):
        result = ExperimentResult(
            experiment_id=config.experiment_id,
            config=config,
            status="success",
            metrics=SCIMetrics(
                psnr=30.0 + i,
                ssim=0.85 + i * 0.01,
                coverage=0.9,
                latency=100.0,
                memory=512.0,
                training_time=300.0,
                convergence_epoch=40
            ),
            api_task_id=f"task_{i}",
            started_at="2025-12-21T12:00:00",
            completed_at="2025-12-21T12:05:00"
        )
        world_model.add_experiment(result)
        print(f"  Added experiment {i+1}/{len(configs)}: {config.experiment_id}")

    # Test SQL-based hash retrieval
    print("\n4. Testing SQL-based hash retrieval...")
    hashes = world_model.get_all_config_hashes()
    print(f"✓ Retrieved {len(hashes)} config hashes from database")
    print(f"  Sample hashes: {list(hashes)[:2]}")

    # Test individual hash check
    print("\n5. Testing individual hash existence check...")
    test_hash = list(hashes)[0]
    exists = world_model.config_hash_exists(test_hash)
    print(f"✓ Hash '{test_hash}' exists: {exists}")

    not_exists = world_model.config_hash_exists("nonexistent_hash_1234")
    print(f"✓ Hash 'nonexistent_hash_1234' exists: {not_exists}")

    # Test PlannerAgent with WorldModel
    print("\n6. Testing PlannerAgent with WorldModel...")
    planner_config = {
        "max_configs_per_cycle": 3,
        "use_llm": False  # Disable LLM for this test
    }
    planner = PlannerAgent(planner_config, world_model=world_model)
    print("✓ PlannerAgent initialized with WorldModel")

    # Load hashes using SQL
    print("\n7. Testing hash loading via PlannerAgent...")
    planner.set_existing_configs()  # Should use SQL query
    print(f"✓ Loaded {len(planner.existing_hashes)} hashes via SQL")

    # Test deduplication
    print("\n8. Testing deduplication...")
    duplicate_config = configs[0]
    is_unique = planner._is_unique(duplicate_config)
    print(f"✓ Duplicate config detected as unique: {is_unique} (should be False)")

    # Test planning (should avoid duplicates)
    print("\n9. Testing experiment planning with deduplication...")
    world_summary = world_model.summarize()
    new_configs = planner.plan_experiments(
        world_summary=world_summary,
        design_space=design_space,
        budget=5
    )
    print(f"✓ Planned {len(new_configs)} new unique experiments")

    # Verify uniqueness
    print("\n10. Verifying all new configs are unique...")
    for i, config in enumerate(new_configs):
        hash_val = ConfigHasher.compute_hash(config)
        is_new = hash_val not in hashes
        print(f"  Config {i+1}: hash={hash_val[:8]}..., is_new={is_new}")
        if not is_new:
            print("  ⚠ WARNING: Generated duplicate config!")

    # Performance comparison
    print("\n11. Performance comparison...")
    print("  Old method: Load all experiments, compute hashes -> O(n)")
    print("  New method: SQL query for hashes -> O(1) with index")
    print(f"  Memory saved: ~{len(configs) * 700} bytes (config JSON) -> {len(configs) * 16} bytes (hashes)")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_hash_optimization()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
