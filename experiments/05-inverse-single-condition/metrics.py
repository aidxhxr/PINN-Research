def stemness(sol, p):
    return (sol["b"] * (1 + p["alpha13"]*sol["h13"])
            / ((1 + sol["apc"]) * (1 + p["alpha5"]*sol["h5"])))
