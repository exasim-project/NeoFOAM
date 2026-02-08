# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for shared file preservation patterns.

Demonstrates:
- Multiple models reading from same file with subdict isolation
- Writing one subdict preserves other sections (critical for shared config files)
"""

import pytest

from foamadapter.io import (
    BaseConfig,
    YAML,
    JSON,
    IOStrategy,
)


# ============================================================================
# Config Classes
# ============================================================================


@IOStrategy(YAML("shared.yaml", subdict="settings.general"))
class GeneralYAMLConfig(BaseConfig):
    timeout: int
    retries: int


@IOStrategy(YAML("shared.yaml", subdict="settings.database"))
class DatabaseYAMLConfig(BaseConfig):
    host: str
    port: int
    name: str


@IOStrategy(YAML("shared.yaml", subdict="settings.cache"))
class CacheYAMLConfig(BaseConfig):
    enabled: bool
    ttl: int


@IOStrategy(JSON("shared.json", subdict="settings.general"))
class GeneralJSONConfig(BaseConfig):
    timeout: int
    retries: int


@IOStrategy(JSON("shared.json", subdict="settings.database"))
class DatabaseJSONConfig(BaseConfig):
    host: str
    port: int
    name: str


@IOStrategy(JSON("shared.json", subdict="settings.cache"))
class CacheJSONConfig(BaseConfig):
    enabled: bool
    ttl: int


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.parametrize(
    "general_class,database_class,cache_class,filename",
    [
        (GeneralYAMLConfig, DatabaseYAMLConfig, CacheYAMLConfig, "shared.yaml"),
        (GeneralJSONConfig, DatabaseJSONConfig, CacheJSONConfig, "shared.json"),
    ],
)
def test_write_preserves_other_subdicts(
    temp_fixture_copy, general_class, database_class, cache_class, filename
):
    """Test that writing one subdict preserves other sections in same file.

    Demonstrates partial file updates - critical for shared config files where
    multiple models manage different sections of the same file.
    """
    test_file = temp_fixture_copy(filename)
    test_dir = test_file.parent

    # Load original values from directory
    original_db = database_class.load(test_dir)
    original_cache = cache_class.load(test_dir)
    assert original_db.host == "localhost"
    assert original_cache.enabled is True

    # Update only general section
    general = general_class.load(test_dir)
    general.timeout = 60
    general.save(test_dir)

    # Verify general changed but other sections unchanged
    updated_general = general_class.load(test_dir)
    updated_db = database_class.load(test_dir)
    updated_cache = cache_class.load(test_dir)

    assert updated_general.timeout == 60  # Changed
    assert updated_general.retries == 3  # Unchanged
    assert updated_db.host == "localhost"  # Unchanged
    assert updated_db.port == 5432  # Unchanged
    assert updated_cache.enabled is True  # Unchanged
    assert updated_cache.ttl == 600  # Unchanged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
