import pickle


def load_solution(path):
    with open(path, 'rb') as fp:
        design, x_trj, u_trj = pickle.load(fp)
    return design, x_trj, u_trj

def save_solution(path, design, x_trj, u_trj):
    solution = [design, x_trj, u_trj]
    with open(path, 'wb') as fp:
        pickle.dump(solution, fp)
