"""Rebuild delivery figures from archived raw records (NumPy/Matplotlib only)."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"03_documents/figures"
def save(fig,name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png","pdf"):fig.savefig(OUT/(name+"."+ext),dpi=220,bbox_inches="tight")
    plt.close(fig)
def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    with (ROOT/"03_documents/tables/epoch_metrics.csv").open(encoding="utf-8-sig") as f:r=list(csv.DictReader(f))
    x=np.array([int(v["epoch"]) for v in r]);a=lambda k:np.array([float(v[k]) for v in r])
    def smooth(y):
        z=np.empty_like(y)
        for lo,hi in ((0,100),(100,len(y))):
            for i in range(lo,hi):z[i]=np.mean(y[max(lo,i-5):min(hi,i+6)])
        return z
    for mode in ("raw","smoothed"):
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
        for ax,tk,dk,title in zip(axes,("train_ce","train_nll"),("dev_nll","dev_nll"),("Sequence-mean training CE and dev token NLL","Token-weighted NLL")):
            # First panel uses sequence-mean dev CE, read directly from source records.
            if tk=="train_ce":
                ev=[json.loads(s) for s in (ROOT/"02_data/training/training.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
                dv=np.array([e["dev"]["mean_sequence_token_nll"] for e in ev if e.get("event")=="epoch_complete"])
                title="Sequence-mean cross entropy"
            else:dv=a(dk)
            for y,label,color in ((a(tk),"Train (online)","#2563a6"),(dv,"Validation","#d15a32")):
                ax.plot(x,y,color=color,alpha=.18 if mode=="smoothed" else 1,lw=.8 if mode=="smoothed" else 1,label=None if mode=="smoothed" else label)
                if mode=="smoothed":ax.plot(x,smooth(y),color=color,lw=2,label=label)
            ax.axvline(100.5,color="gray",ls="--");ax.set(title=title,xlabel="Epoch",ylabel="Nats");ax.grid(alpha=.18);ax.legend()
        fig.suptitle("Two-stage optimization: LR 0.003 (epochs 1-100), LR 0.001 (epochs 101-300)")
        save(fig,"training_loss_"+mode)
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
        for ax,keys,title in ((axes[0],("train_accuracy","dev_accuracy"),"Teacher-forced token accuracy"),(axes[1],("dev_sequence_accuracy",),"Validation whole-sequence accuracy (gold prefixes)")):
            for k in keys:
                y=a(k)*100;color="#2563a6" if k.startswith("train") else "#d15a32"
                label="Train (online)" if k.startswith("train") else "Validation"
                ax.plot(x,y,color=color,alpha=.18 if mode=="smoothed" else 1,lw=.8 if mode=="smoothed" else 1,label=None if mode=="smoothed" else label)
                if mode=="smoothed":ax.plot(x,smooth(y),color=color,lw=2,label=label)
            ax.axvline(100.5,color="gray",ls="--");ax.set(title=title,xlabel="Epoch",ylabel="Accuracy (%)",ylim=(0,102));ax.grid(alpha=.18);ax.legend()
        save(fig,"training_accuracy_"+mode)
    p=ROOT/"02_data/pretraining"
    meta=json.loads((p/"paper.json").read_text(encoding="utf-8"))
    z=np.load(p/"paper.npz",allow_pickle=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for ax,kind in zip(axes,("softmax","sigmoid")):
        y=z[kind+"_loss_history"];xx=np.arange(1,len(y)+1)
        ax.plot(xx,y,color="#2563a6",alpha=.3,lw=.6,label="Raw training loss")
        avg=np.array([np.mean(y[max(0,i-15):min(len(y),i+16)]) for i in range(len(y))])
        ax.plot(xx,avg,color="#2563a6",lw=1.5,label="31-step moving mean")
        ax.axhline(float(z[kind+"_validation_loss"]),color="#d15a32",ls="--",label="Final validation loss")
        ax.set(title=kind.capitalize()+" pretraining",xlabel="Update",ylabel="1 - state fidelity",yscale="log");ax.grid(alpha=.18);ax.legend(fontsize=8)
    save(fig,"nonlinear_pretraining")
    fig,ax=plt.subplots(figsize=(11,4));ax.set_xlim(0,11);ax.set_ylim(0,4);ax.axis("off")
    boxes=[(.1,2.7,2.4,.8,"Source tokens + target prefix"),
           (3,2.7,2.4,.8,"Embedding + RX position"),
           (6,2.7,2.4,.8,"Shared Q / K / V maps"),
           (6,1.25,4.8,.85,"Per-context Pair: SwapTest + entangler\nthen final swap=0 probability row"),
           (.1,1.25,4.8,.85,"Classical weighted probability aggregation\nand state re-preparation"),
           (.1,.05,7,.65,"Quantum Softmax -> GW1 -> Sigmoid -> GW2"),
           (8,.05,2.8,.65,"8-basis readout\nnext token / EOS")]
    for x0,y0,w,h,t in boxes:
        ax.add_patch(FancyBboxPatch((x0,y0),w,h,boxstyle="round,pad=0.04",fc="#eff5fc",ec="#2563a6",lw=1.2))
        ax.text(x0+w/2,y0+h/2,t,ha="center",va="center",fontsize=9)
    for p1,p2 in [((2.5,3.1),(3,3.1)),((5.4,3.1),(6,3.1)),((7.2,2.7),(7.2,2.1)),((6,1.67),(4.9,1.67)),((2.5,1.25),(2.5,.7)),((7.1,.37),(8,.37))]:
        ax.annotate("",xy=p2,xytext=p1,arrowprops=dict(arrowstyle="->",lw=1.4,color="#444444"))
    ax.text(9.65,3.15,"A1F1E3\n3-qubit token",ha="center",va="center",fontsize=10)
    save(fig,"serial_architecture")
    print("Figures generated:",OUT)
if __name__=="__main__":main()
