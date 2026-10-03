"""Apply recorded primary-source corrections after the publisher audit."""
import json
from pathlib import Path
import re


if __name__ == "__main__":
    bibliography = Path("paper/references.bib")
    manuscript = Path("paper/main.tex")
    content, source = bibliography.read_text(encoding="utf-8"), manuscript.read_text(encoding="utf-8")
    report_path = Path("results/research_checks/reference_metadata.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    overrides = json.loads(Path("scripts/reference_primary_overrides.json").read_text(encoding="utf-8"))
    records = {record["key"]: record for record in report["records"]}
    for match in list(re.finditer(r'@(\w+)\{([^,]+),(.*?)(?=\n@|\Z)', content, re.S)):
        kind, key = match[1], match[2]
        if key not in overrides:
            continue
        correction = overrides[key]
        fields = dict(re.findall(r'(\w+)\s*=\s*\{(.*?)\}(?=,?\s*\n)', match[3], re.S))
        for field in correction.get("remove", []):
            fields.pop(field, None)
        fields.update(correction["fields"])
        fields["url"] = correction["source"]
        new_key = correction.get("key", key)
        if new_key != key:
            source = re.sub(r'(?<=[{,])' + re.escape(key) + r'(?=[,}])', new_key, source)
        updated = "@" + correction.get("kind", kind) + "{" + new_key + ",\n" + ",\n".join(
            "  " + name + " = {" + value + "}" for name, value in fields.items()) + "\n}\n"
        content = content.replace(match[0], updated)
        record = records[key]
        record.update(key=new_key, status="primary_source_verified", primary_source=correction["source"],
                      verified_metadata=fields, verification_note=correction.get("note", "primary bibliographic metadata checked"))
    content = content.replace("d’Avila", "d'Avila")
    bibliography.write_text(content, encoding="utf-8", newline="\n")
    manuscript.write_text(source, encoding="utf-8", newline="\n")
    report["unresolved"] = [record["key"] for record in report["records"] if record["status"] not in {"publisher_metadata_matched", "primary_source_verified"}]
    report["verified_entries"] = len(report["records"]) - len(report["unresolved"])
    report["audit_scope_note"] = "Metadata audit of all 28 cited entries. Unsupported Hilario pages removed; unavailable private Wang draft replaced with published second edition. Scientific claim support remains a separate review."
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(verified=report["verified_entries"], unresolved=report["unresolved"])))
