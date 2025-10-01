#!/usr/bin/env python3
"""Test script to verify groups_read_mode column exists and works"""

import os
import subprocess
import sys

def test_groups_read_mode():
    # Check if DATABASE_URL is set
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ DATABASE_URL environment variable not set")
        print("\n⚠️  Set it first:")
        print("   export DATABASE_URL='your-postgres-url'")
        return False

    print("✓ DATABASE_URL is set")

    # Try to query the column using psql
    query = "SELECT name, groups_read_mode FROM agents LIMIT 5;"

    try:
        result = subprocess.run(
            ['psql', db_url, '-c', query],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print("✓ groups_read_mode column exists")
            print("✓ Sample data:")
            print(result.stdout)
            return True
        else:
            # Check if error is about missing column
            if 'column "groups_read_mode" does not exist' in result.stderr:
                print("❌ groups_read_mode column does NOT exist")
                print("\n⚠️  You need to run the migration:")
                print("   psql $DATABASE_URL -f migrations/convert_cross_group_to_tristate.sql")
            else:
                print(f"❌ Error querying database: {result.stderr}")
            return False

    except FileNotFoundError:
        print("❌ psql command not found")
        print("\n⚠️  Install PostgreSQL client or run migration manually")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Database query timed out")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_groups_read_mode()
    sys.exit(0 if success else 1)
