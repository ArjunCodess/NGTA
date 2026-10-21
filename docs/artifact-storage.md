# Artifact storage

The current branch excludes 297 generated estimator/inference files, totaling 8,691,914,602 bytes. Their original local files remain at the paths recorded in [the archive manifest](../results/research_checks/artifact_archive_manifest.json). Ignore rules prevent accidentally adding them again. Code, configurations, source manifests, small results, figures and replay instructions remain tracked.

Verify a retained copy with:

```powershell
python scripts/verify_local_artifacts.py
```

For an archive with the same directory layout, pass `--root` with its location. Every file must match both its saved size and SHA256. All local files were reverified on 2026-10-20. A retained hard-link recovery set points to the same file contents on C:, and a verified old-history Git bundle is stored separately. Neither the working files nor same-volume hard links are an independent physical artifact backup; free disk capacity prevented a complete second copy. Copy the artifacts to approved storage before removing this checkout or changing them in place. Choose storage and sharing permissions that match the original data's access terms.

A new clone will not contain the removed preprocessors, fitted estimators or cached inference arrays. The README training commands can regenerate studies; exact saved-run replay requires restoring the retained archive. No archive download URL is published because no external archive has been provisioned.

On 2026-10-20 the 297 paths were also removed from the published `version-2` history, with current branch-tip contents unchanged. The 220 distinct historical LFS object IDs and verification are in [the purge report](../results/research_checks/github_purge/verification.json). GitHub still controls orphaned LFS objects and cached/PR references; remote purge and quota reclamation are not yet confirmed. See [push and purge status](push-status.md) and [GitHub's removal instructions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-git-large-file-storage).

The source-only technical release is built with `python scripts/package_reproducibility.py`; it contains no fitted archive, patient-level traces or datasets. The exact 36-fit regeneration recipe and private output/disk checks are in [reproducibility.md](reproducibility.md). The current saved study directories total about 10.14 GB, so this disk does not have room for a duplicate full regeneration plus reserve. Verification can reuse the original files without making another copy.
