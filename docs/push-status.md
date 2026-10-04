# Push status

The destination is `https://github.com/ArjunCodess/NGTA.git`, branch `version-2`. All work remains on that branch.

The owner published the earlier commits through `4209d8e`. The previous automatic-review rejection is no longer a branch-publication blocker. The ordinary removal commit `738b690` was subsequently pushed successfully to the same branch.

That commit removes 287 hospital-study fitted estimators/inference arrays and ten public sensitivity inference arrays from the current tree. All 297 local files remain intact: their sizes and SHA256 hashes match the [archive manifest](../results/research_checks/artifact_archive_manifest.json). Ignore rules prevent these generated files from being added again by ordinary staging. Existing source datasets, checkpoints and trace tables were outside this requested removal.

An ordinary removal commit does not erase earlier commits or remote LFS objects, and does not reclaim their GitHub storage quota. Permanent removal is a separate history-cleanup and remote-storage operation. [GitHub's removal instructions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-git-large-file-storage) explain that old LFS objects remain even after history cleanup and describe contacting support when repository deletion is unsuitable. No history rewrite, force push or repository deletion was performed.

New clones need an approved separate archive for exact replay of the removed artifacts, or must regenerate them using the training commands. The retained working directory is not a separate backup. See [artifact storage](artifact-storage.md).
