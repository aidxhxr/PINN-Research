"""First-poster dynamics, saved training diagnostics and presentation marginals."""
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

BLUE, ORANGE, GARNET, INK, GREY = "#1f77b4", "#e88028", "#78253b", "#152a3a", "#7f7f7f"
STATES = [("m", r"MYC $m$", BLUE), ("p", r"APC $p$", ORANGE)]


def save(fig, out, name):
    fig.savefig(out/f"{name}.pdf")
    fig.savefig(out/f"{name}.png", dpi=110)
    plt.close(fig)


def trajectory_pair(data, out, *, fitted):
    d = np.load(data/"normal_treatment_comparison.npz")
    states = d["states"].tolist()
    width, height = 14.6, 4.35
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharey=True)
    fig.subplots_adjust(left=.075, right=.97, bottom=.31, top=.74, wspace=.17)
    for ci, ax in enumerate(axes):
        for state, label, color in STATES:
            k = states.index(state)
            ax.plot(d["t"], d["reference"][ci,:,k], color=color,
                    lw=2.6 if not fitted else 3.8, alpha=1 if not fitted else .38)
            if fitted:
                ax.plot(d["t"], d["pinn"][ci,:,k], color=color, lw=1.8, ls=(0,(5,3)))
        ax.set(xlim=(0,150), ylim=(.20,1.20), xticks=[0,50,100,150], yticks=[.4,.8,1.2])
        ax.tick_params(labelsize=22, length=4)
        ax.grid(False)
        ax.spines[["top","right"]].set_visible(False)
        pos=ax.get_position()
        title = r"No treatment: $D_R=0$" if ci==0 else r"ATRA treatment: $D_R=1.5$"
        fig.text(pos.x0, .915, title, fontsize=27, va="center")
    axes[0].set_ylabel("concentration", fontsize=24, labelpad=8)
    fig.text(.535,.18,r"dimensionless time $\tau$",ha="center",va="center",fontsize=23)

    handles=[Line2D([],[],color=c,lw=2.7,label=label) for _,label,c in STATES]
    if fitted:
        handles += [Line2D([],[],color=GREY,lw=3,alpha=.5,label="Radau"),
                    Line2D([],[],color=INK,lw=1.8,ls=(0,(5,3)),label="PINN")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.535,-.015),
               ncol=len(handles),frameon=False,fontsize=22,handlelength=1.6,
               columnspacing=1.4,handletextpad=.5)
    name="normal_pinn" if fitted else "normal_reference"
    pos=axes[1].get_position()
    arrow={"left":width*(pos.x0+pos.width*40/150),
           "right":width*(pos.x0+pos.width*88/150),"y":height*pos.y1+.11}
    save(fig,out,name)
    tex=["% Generated; the ATRA annotation is editable LaTeX.",
         r"\begingroup\begin{tikzpicture}[x=1in,y=1in]",
         r"\node[anchor=south west,inner sep=0pt] at (0,0)"
         rf" {{\includegraphics[width=14.6in]{{assets/{name}.pdf}}}};",
         r"\draw[{Stealth[length=8pt,width=7pt]}-{Stealth[length=8pt,width=7pt]},"
         r"draw=Ink!80,line width=1.7pt]"
         f" ({arrow['left']:.5f},{arrow['y']:.5f}) --"
         r" node[above=4pt,font=\fontsize{22}{27}\selectfont,text=Ink] {ATRA}"
         f" ({arrow['right']:.5f},{arrow['y']:.5f});",
         r"\end{tikzpicture}\endgroup"]
    (out/f"{name}-window.tex").write_text("\n".join(tex)+"\n")
    (out/f"{name}_audit.json").write_text(json.dumps({"condition_order":d["conditions"].tolist(),
        "DR":d["DR"].tolist(),"states":[s for s,_,_ in STATES],"normal_only":True,
        "grid":False,"background":"white","ATRA_arrow":arrow,
        "inset":False},indent=2)+"\n")


def training_diagnostics(data,out):
    h=json.loads((data/"normal_inverse_history.json").read_text())
    truth=json.loads((data/"normal_inverse_recovered.json").read_text())["true"]
    epochs=np.array(h["epoch"])
    fig,axes=plt.subplots(1,2,figsize=(14.6,3.65))
    fig.subplots_adjust(left=.08,right=.965,bottom=.32,top=.83,wspace=.27)
    curves=[("Data",np.array(h["Ld"]),BLUE),
            ("Physics",np.array(h["lam_phys"])*np.array(h["Lp"]),GARNET),
            ("Initial state",20*np.array(h["Lic"]),ORANGE)]
    for label,y,color in curves:
        axes[0].semilogy(epochs,y,color=color,lw=2,marker="o",ms=3.5,label=label)
    assert np.allclose(sum(y for _,y,_ in curves),h["loss"])
    for key,label,color in [("W",r"$W$",BLUE),("thetaP",r"$\theta_P$",ORANGE)]:
        err=100*np.abs(np.array(h[key])-truth[key])/abs(truth[key])
        axes[1].plot(epochs,err,color=color,lw=2,marker="o",ms=3.5,label=label)
    axes[1].axhline(10,color=GREY,lw=1,ls=(0,(4,3)))
    axes[1].text(1970,15,"10%",color=GREY,fontsize=18,ha="right")
    for ax,title in zip(axes,["Training loss components","Selected parameter errors"]):
        ax.set(xlim=(0,2000),xticks=[0,1000,2000])
        ax.set_title(title,fontsize=25,pad=10,loc="left")
        ax.tick_params(labelsize=22,length=4)
        ax.grid(False)
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.37),ncol=3,
                  fontsize=21,frameon=False,handlelength=1.25,
                  columnspacing=1,handletextpad=.4,borderaxespad=0)
    axes[0].set_ylabel("weighted loss",fontsize=22)
    axes[0].set_yticks([1e0,1e-3,1e-6])
    axes[1].set_ylabel("relative error (%)",fontsize=22)
    axes[1].set_ylim(0,55)
    axes[1].set_yticks([0,25,50])
    fig.text(.535,.135,"Adam epoch",ha="center",fontsize=23)
    save(fig,out,"normal_training")


def bayesian_marginals(data,out):
    panels=[("normal","Normal",["etaBM","kappaBM","kappaR"],BLUE),
            ("severe","Severe APC loss",["etaBM","aM","lambdaP"],GARNET)]
    labels={"etaBM":r"$\eta_{BM}$","kappaBM":r"$\kappa_{BM}$", "kappaR":r"$\kappa_R$",
            "aM":r"$a_M$","lambdaP":r"$\lambda_P$"}
    fig,axes=plt.subplots(2,3,figsize=(14.6,8.0))
    fig.subplots_adjust(left=.075,right=.97,bottom=.14,top=.865,wspace=.28,hspace=.85)
    audit=[]
    for row,(slug,title,params,color) in enumerate(panels):
        d=np.load(data/f"presentation_{slug}_posterior.npz")
        summary=json.loads((data/f"presentation_{slug}_posterior_summary.json").read_text())
        names=d["names"].tolist()
        fig.text(.075,.995 if row==0 else .515,title,fontsize=26,color=color,
                 fontweight="bold",va="top")
        for ax,key in zip(axes[row],params):
            k=names.index(key); values=d["values"][:,k]; true=float(d["true"][k])
            mean=float(values.mean())
            assert np.isclose(mean,summary["params"][key]["post_mean"])
            ax.hist(values,bins=40,density=True,color=color,alpha=.72,linewidth=0)
            ax.axvline(true,color=INK,lw=2.2)
            ax.axvline(mean,color=INK,lw=1.6,ls=(0,(4,3)))
            ax.set_title(labels[key],fontsize=28,pad=5)
            ax.set_ylim(bottom=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
            formatter=ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.xaxis.set_major_formatter(formatter)
            ax.tick_params(labelsize=20,length=3)
            ax.grid(False)
            audit.append({"regime":slug,"parameter":key,"draws":len(values),
                          "truth":true,"mean":mean,"ess":summary["params"][key]["ess"]})
        axes[row,0].set_ylabel("density",fontsize=23)
    fig.legend(handles=[Line2D([],[],color=INK,lw=2.2,label="Truth"),
                        Line2D([],[],color=INK,lw=1.6,ls=(0,(4,3)),label="Posterior mean")],
               loc="lower center",bbox_to_anchor=(.54,-.02),ncol=2,
               frameon=False,fontsize=23,handlelength=2,columnspacing=2)
    save(fig,out,"presentation_marginals")
    (out/"presentation_marginals_audit.json").write_text(json.dumps(audit,indent=2)+"\n")


def make_normal_inference(data,out):
    with plt.rc_context({"axes.grid":False,"axes.facecolor":"white","figure.facecolor":"white"}):
        trajectory_pair(data,out,fitted=False)
        trajectory_pair(data,out,fitted=True)
        training_diagnostics(data,out)
        bayesian_marginals(data,out)
