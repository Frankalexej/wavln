GOOD_RUNS = {
    "b": [2, 3, 5, 7, 9], 
    "u": [1, 2, 6, 7, 9]
}

def stringify(runs, deli=""): 
    return deli.join([str(run) for run in runs])