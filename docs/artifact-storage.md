# Artifact storage

The current branch excludes 297 generated estimator/inference files, totaling 8,691,914,602 bytes. Their original local files remain at the paths recorded in [the archive manifest](../results/research_checks/artifact_archive_manifest.json). Ignore rules prevent accidentally adding them again. Code, configurations, source manifests, small results, figures and replay instructions remain tracked.

Verify a retained copy with:

```powershell
python scripts/verify_local_artifacts.py
```

For an archive with the same directory layout, pass `--root` with its location. Every file must match both its saved size and SHA256. Local working files are not a separate backup; copy them to an approved archive before removing this checkout. Choose storage and sharing permissions that match the original data's access terms.

A new clone will not contain the removed preprocessors, fitted estimators or cached inference arrays. The README training commands can regenerate studies; exact saved-run replay requires restoring the retained archive. No archive download URL is published because no external archive has been provisioned.

The removal commit changes the branch tip, not older commits. GitHub documents that removed LFS objects remain in remote storage and count toward its quota. Purging Git history requires a separate rewrite, and purging remote objects may require GitHub Support. See [GitHub's removal instructions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-git-large-file-storage). The repository was not deleted or recreated.
