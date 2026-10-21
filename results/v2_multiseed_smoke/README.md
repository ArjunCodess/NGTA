# Five-seed integration runs only

Seeds 0 through 4 each train for two epochs with d_model=8 and three MC passes on the same TCGA case partitions. These runs verify repeated training, fixed cohort splitting, per-seed replay, and actual variability aggregation. They are not a full clinical confirmation experiment; the TCGA held-out partitions still contain no recorded genomic positives.

The aggregate tables are under `submission`. Each seed preserves its own checkpoint, preprocessing, source lineage, MC passes, and events. `results/v2_validation/integration_checks.json` records that all split-ID hashes agree and all five replays pass.
