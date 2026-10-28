"""Portable entry point; numerical model source remains unchanged."""
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent/"model"))
def verified_dataset_identifier(data_root):
    root=Path(data_root).resolve()
    index=json.loads((root/"delivery_index.json").read_text(encoding="utf-8"))
    for name,expected in index["files"].items():
        path=(root/name).resolve()
        if path.parent!=root or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Delivered dataset integrity failure: {name}")
    return index["source_dataset_sha256"]
def main():
    import train_scan_jax_impl as trainer
    import train_dyck_jax
    # Explicit archival compatibility adapter: verify the supplied split bytes,
    # then retain the original experiment's identifier. It does not claim to
    # recompute a full source-directory digest.
    trainer.dataset_sha256=verified_dataset_identifier
    argv=sys.argv[1:]
    if argv and argv[0]!="pretrain-nonlinear":
        if "--data-root" not in argv:argv+=["--data-root",str(ROOT/"02_data/dataset/DyckLanguage")]
        if "--nonlinear-artifact" not in argv:argv+=["--nonlinear-artifact",str(ROOT/"02_data/pretraining/paper")]
        if argv[0]=="evaluate":
            if "--checkpoint" not in argv:argv+=["--checkpoint",str(ROOT/"02_data/training/best")]
            if "--split" not in argv:argv+=["--split","test"]
            if "--split" in argv and argv[argv.index("--split")+1] not in ("train","dev","test"):
                raise ValueError("This delivery supports train/dev/test only.")
    return train_dyck_jax.main(argv)
if __name__=="__main__":raise SystemExit(main())
