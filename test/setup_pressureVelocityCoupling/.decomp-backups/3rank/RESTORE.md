# Restore 3-rank decomposition

The 2-rank decomposition under setup_pressureVelocityCoupling was created for the
Phase 2 LSA-01 spike (plan 02-01). Other distributed tests still expect 3 ranks.

To restore the 3-rank decomposition:

```bash
cd test/setup_pressureVelocityCoupling
rm -rf processor0 processor1
mv .decomp-backups/3rank/processor0 .decomp-backups/3rank/processor1 .decomp-backups/3rank/processor2 .
cp .decomp-backups/3rank/decomposeParDict.original system/decomposeParDict
```

After restore, the 3-rank distributed tests pass as before.

Long-term: Phase 5 will add stable side-by-side decomposition directories (or move setups
into per-rank subdirectories) so this swap is no longer required.
