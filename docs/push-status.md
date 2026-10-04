# Push status

The requested destination is `https://github.com/ArjunCodess/NGTA.git`, branch `version-2`. All work remains on that branch.

The explicit LFS upload completed successfully: 210 unique objects, approximately 7.6 GB, backing 287 tracked hospital-study artifact paths. The following branch push was rejected by automatic approval review before execution. Its stated reason was that those clinical-derived artifacts are potentially sensitive and require explicit authorization naming both the payload and destination. General instructions to push were not accepted by that review.

The remote branch has not been advanced by this attempt. The upload and branch update are separate operations; an uploaded LFS object is not evidence that the commits were pushed. No alternate remote, history rewrite, hook bypass or indirect push was attempted.

The [payload provenance audit](../results/research_checks/push_payload_audit.json) identifies the 287 artifacts and their existing WiDS source. All additional public sensitivity inputs and outputs are documented in [source evidence](source-evidence.md), with independent clinical eligibility explicitly false.

Subsequent local work adds ten LFS inference-cache arrays from the openly available PhysioNet sensitivity. Those ten objects have not been uploaded; the final branch now has 297 tracked LFS artifact paths. Source access, units, unavailable inputs and eligibility limits are documented with their publisher links.

To unblock the requested branch publication, explicitly approve publishing the 287 clinical-derived hospital-study model/array artifacts, ten public PhysioNet inference-cache arrays and the committed NGTA work to `https://github.com/ArjunCodess/NGTA.git` on `version-2`. This approval requirement comes from automatic approval review, not from Git or a missing authentication credential. Authentication and branch-update success remain untested because execution was rejected.
