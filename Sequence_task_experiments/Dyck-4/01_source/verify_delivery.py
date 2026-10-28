"""Verify archived records, dataset semantics, and delivery file hashes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ledger=json.loads((ROOT/"source_records.json").read_text(encoding="utf-8"))
    for row in ledger["records"]:assert sha(ROOT/row["path"])==row["sha256"],row["path"]
    manifest=ROOT/"delivery_manifest.json"
    if manifest.exists():
        for row in json.loads(manifest.read_text(encoding="utf-8"))["files"]:assert sha(ROOT/row["path"])==row["sha256"],row["path"]
    dataset=ROOT/"02_data/dataset/DyckLanguage";seen=set()
    for split,count in (("train",210),("dev",49),("test",49)):
        rows=[json.loads(s) for s in (dataset/(split+".jsonl")).read_text(encoding="utf-8").splitlines() if s.strip()]
        assert len(rows)==count
        assert (dataset/(split+".src")).read_text(encoding="utf-8").splitlines()==[r["input"] for r in rows]
        assert (dataset/(split+".tgt")).read_text(encoding="utf-8").splitlines()==[r["output"] for r in rows]
        for row in rows:
            assert row["input"] not in seen;seen.add(row["input"])
            stack=[];pairs=dict(zip("([{<",")]}>"))
            for token in row["input"].split():
                if token in pairs:stack.append(pairs[token])
                else:assert stack and stack.pop()==token
            assert list(reversed(stack))==row["output"].split()
    expected=json.loads((ROOT/"02_data/training/best.json").read_text(encoding="utf-8"))["metadata"]["nonlinear_artifact_sha256"]
    assert sha(ROOT/"02_data/pretraining/paper.npz")==expected
    print("PASS: source records, dataset alignment/semantics, pretraining provenance and delivery hashes.")
if __name__=="__main__":main()
