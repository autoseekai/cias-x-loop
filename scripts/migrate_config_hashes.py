#!/usr/bin/env python3
"""
Migration Script: Add Config Hashes to Existing Database

This script updates existing experiments in the database by computing
and storing their config hashes. This is needed for databases created
before the config_hash optimization was implemented.
"""

import sqlite3
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import hashlib

from loguru import logger


def compute_hash_from_dict(config_dict: Dict[str, Any]) -> str:
    """
    Compute hash from a configuration dictionary
    (Standalone version to avoid circular imports)

    Args:
        config_dict: Configuration dictionary

    Returns:
        SHA256 hash string
    """
    hashable_dict = {
        "forward": config_dict.get("forward_config", {}),
        "recon": {
            "family": config_dict.get("recon_family", ""),
            **config_dict.get("recon_params", {})
        },
        "uq": {
            "scheme": config_dict.get("uq_scheme", ""),
            "params": config_dict.get("uq_params", {})
        },
        "train": config_dict.get("train_config", {})
    }
    json_str = json.dumps(hashable_dict, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def check_schema(db_path: str) -> bool:
    """
    Check if the database has the config_hash column

    Args:
        db_path: Path to database

    Returns:
        True if column exists, False otherwise
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(experiments)")
    columns = [row[1] for row in cursor.fetchall()]

    conn.close()
    return "config_hash" in columns


def add_config_hash_column(db_path: str):
    """
    Add config_hash column and index to database

    Args:
        db_path: Path to database
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Add column
        cursor.execute("""
            ALTER TABLE experiments ADD COLUMN config_hash TEXT
        """)
        logger.info("Added config_hash column to experiments table")

        # Add index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_config_hash
            ON experiments(config_hash)
        """)
        logger.info("Created index on config_hash column")

        conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            logger.info("config_hash column already exists")
        else:
            raise
    finally:
        conn.close()


def migrate_hashes(db_path: str, batch_size: int = 100, dry_run: bool = False) -> Dict[str, int]:
    """
    Compute and store config hashes for existing experiments

    Args:
        db_path: Path to database
        batch_size: Number of records to process in each batch
        dry_run: If True, don't actually update the database

    Returns:
        Statistics dictionary
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Count experiments without hash
    cursor.execute("""
        SELECT COUNT(*) FROM experiments WHERE config_hash IS NULL
    """)
    total_to_migrate = cursor.fetchone()[0]

    if total_to_migrate == 0:
        logger.info("No experiments need migration")
        return {"total": 0, "migrated": 0, "errors": 0}

    logger.info(f"Found {total_to_migrate} experiments without config_hash")

    # Process in batches
    migrated = 0
    errors = 0
    offset = 0

    while True:
        # Fetch batch
        cursor.execute("""
            SELECT experiment_id, config_json
            FROM experiments
            WHERE config_hash IS NULL
            LIMIT ? OFFSET ?
        """, (batch_size, offset))

        batch = cursor.fetchall()
        if not batch:
            break

        for exp_id, config_json in batch:
            try:
                config_dict = json.loads(config_json)
                hash_val = compute_hash_from_dict(config_dict)

                if not dry_run:
                    cursor.execute("""
                        UPDATE experiments
                        SET config_hash = ?
                        WHERE experiment_id = ?
                    """, (hash_val, exp_id))

                migrated += 1

                if migrated % 100 == 0:
                    logger.info(f"Progress: {migrated}/{total_to_migrate} experiments")

            except Exception as e:
                logger.error(f"Error processing experiment {exp_id}: {e}")
                errors += 1

        if not dry_run:
            conn.commit()

        offset += batch_size

    conn.close()

    stats = {
        "total": total_to_migrate,
        "migrated": migrated,
        "errors": errors
    }

    return stats


def verify_migration(db_path: str) -> Dict[str, int]:
    """
    Verify migration results

    Args:
        db_path: Path to database

    Returns:
        Statistics dictionary
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM experiments")
    total_experiments = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM experiments WHERE config_hash IS NOT NULL")
    with_hash = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM experiments WHERE config_hash IS NULL")
    without_hash = cursor.fetchone()[0]

    # Check for duplicate hashes
    cursor.execute("""
        SELECT config_hash, COUNT(*) as count
        FROM experiments
        WHERE config_hash IS NOT NULL
        GROUP BY config_hash
        HAVING count > 1
    """)
    duplicates = cursor.fetchall()

    conn.close()

    return {
        "total_experiments": total_experiments,
        "with_hash": with_hash,
        "without_hash": without_hash,
        "duplicate_hashes": len(duplicates),
        "duplicates": duplicates[:10] if duplicates else []  # Show first 10
    }


def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(
        description="Migrate existing database to include config hashes"
    )
    parser.add_argument(
        "database",
        type=str,
        help="Path to SQLite database file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually updating database"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of records to process in each batch (default: 100)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify migration status without performing migration"
    )

    args = parser.parse_args()

    # Check database exists
    if not Path(args.database).exists():
        logger.error(f"Database not found: {args.database}")
        return 1

    logger.info(f"Processing database: {args.database}")

    # Verify only mode
    if args.verify_only:
        logger.info("Running verification...")
        stats = verify_migration(args.database)
        logger.info(f"Total experiments: {stats['total_experiments']}")
        logger.info(f"With hash: {stats['with_hash']}")
        logger.info(f"Without hash: {stats['without_hash']}")
        if stats['duplicate_hashes'] > 0:
            logger.warning(f"Found {stats['duplicate_hashes']} duplicate hashes:")
            for hash_val, count in stats['duplicates']:
                logger.warning(f"  {hash_val}: {count} experiments")
        return 0

    # Check and add schema if needed
    if not check_schema(args.database):
        logger.info("config_hash column not found, adding...")
        add_config_hash_column(args.database)
    else:
        logger.info("config_hash column exists")

    # Perform migration
    mode_str = "DRY RUN" if args.dry_run else "MIGRATION"
    logger.info(f"Starting {mode_str}...")

    stats = migrate_hashes(
        args.database,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )

    # Report results
    logger.info(f"\n{'='*60}")
    logger.info(f"{mode_str} COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total experiments to migrate: {stats['total']}")
    logger.info(f"Successfully processed: {stats['migrated']}")
    logger.info(f"Errors: {stats['errors']}")

    if args.dry_run:
        logger.info("\nThis was a dry run. No changes were made to the database.")
        logger.info("Run without --dry-run to apply changes.")
    else:
        # Verify
        logger.info("\nVerifying migration...")
        verify_stats = verify_migration(args.database)
        logger.info(f"Experiments with hash: {verify_stats['with_hash']}/{verify_stats['total_experiments']}")

        if verify_stats['without_hash'] == 0:
            logger.success("✓ All experiments now have config hashes!")
        else:
            logger.warning(f"⚠ {verify_stats['without_hash']} experiments still missing hashes")

        if verify_stats['duplicate_hashes'] > 0:
            logger.info(f"\nFound {verify_stats['duplicate_hashes']} duplicate configurations:")
            for hash_val, count in verify_stats['duplicates']:
                logger.info(f"  {hash_val}: {count} experiments")

    return 0


if __name__ == "__main__":
    exit(main())
