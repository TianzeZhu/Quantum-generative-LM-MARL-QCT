"""Reconstruct the synthetic sample stream; do not retrain or overwrite weights."""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent/"model"))
import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import numpy as np
from nonlinear_pretrain_jax import random_nonnegative_states,target_states,fidelity_loss

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--artifact-dir",type=Path,default=ROOT/"02_data/pretraining");args=ap.parse_args()
    folder=args.artifact_dir
    meta=json.loads((folder/"paper.json").read_text(encoding="utf-8"));cfg=meta["config"]
    assert jax.__version__==meta["metadata"]["environment"]["jax"],"Use the recorded JAX version."
    assert hashlib.sha256((folder/"paper.npz").read_bytes()).hexdigest()==meta["npz_sha256"]
    artifact=np.load(folder/"paper.npz",allow_pickle=False)
    result=dict(origin="deterministic_reconstruction_not_original_capture",jax=jax.__version__,backend=jax.default_backend(),dtype="float64",source_artifact_sha256=meta["npz_sha256"],config=cfg,functions={})
    for kind in ("softmax","sigmoid"):
        key=jax.random.PRNGKey(cfg["seed"]+(kind=="sigmoid"))
        _,key=jax.random.split(key)
        def one_step(key,_):
            key,sample_key=jax.random.split(key)
            inputs=random_nonnegative_states(sample_key,cfg["batch_size"],cfg["n"])
            targets=target_states(inputs,kind,cfg["softmax_beta"])
            return key,(inputs,targets)
        key,(inputs,targets)=jax.jit(lambda k:jax.lax.scan(one_step,k,None,length=cfg["steps"]))(key)
        _,validation_key=jax.random.split(key)
        vi=random_nonnegative_states(validation_key,cfg["validation_size"],cfg["n"])
        vt=target_states(vi,kind,cfg["softmax_beta"])
        observed=float(jax.jit(fidelity_loss,static_argnums=(3,4))(jnp.asarray(artifact[kind+"_params"]),vi,vt,cfg["layers1"],cfg["layers2"]))
        expected=float(artifact[kind+"_validation_loss"])
        assert np.isclose(observed,expected,atol=1e-7,rtol=1e-7),(kind,observed,expected)
        path=folder/(kind+"_reconstructed_samples.npz")
        np.savez_compressed(path,train_inputs=np.asarray(inputs),train_targets=np.asarray(targets),validation_inputs=np.asarray(vi),validation_targets=np.asarray(vt))
        result["functions"][kind]=dict(training_samples=cfg["steps"]*cfg["batch_size"],validation_samples=cfg["validation_size"],reconstructed_validation_loss=observed,recorded_validation_loss=expected,absolute_error=abs(observed-expected),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (folder/"reconstruction_metadata.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
