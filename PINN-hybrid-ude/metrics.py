def stemness(sol, p):
    return (sol["b"] * (1 + p["alpha13"]*sol[r"$h_{13}$"])
            / ((1 + sol["p"]) * (1 + p["alpha5"]*sol[r"$h_5$"])))
