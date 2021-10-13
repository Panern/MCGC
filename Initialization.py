
import warnings
import numpy as np
warnings.filterwarnings("ignore")


def Initail_S(H_v,av,a,gama):

    I = np.eye(len(H_v[0]))
    A = av[0] + I
    D = np.sum(A, axis=1)
    D = np.power(D, -0.5)
    D[np.isinf(D)] = 0
    D = np.diagflat(D)
    A = D.dot(A).dot(D)
    # kk =len( np.unique(gnd))
    Ls = D - A

    num_view = len(H_v)
    nada = [1 / num_view for i in range(num_view)]


    print("Initialization!")
    # S = np.zeros((3025,3025))
    print('Begin!\n')
    for i in range(3):
        XtX_bar = 0
        for j in range(num_view):
            XtX_bar = XtX_bar + nada[j] * H_v[j].dot(H_v[j].T)
        tmp = np.linalg.inv(a * I + XtX_bar)
        S = tmp.dot(a * Ls + XtX_bar)
        # S = tmp.dot(XtX_bar)
        for j in range(num_view):
            nada[j] = (-((np.linalg.norm(H_v[j].T - (H_v[j].T).dot(S))) ** 2 + a * (
                np.linalg.norm(S - Ls)) ** 2) / gama) ** (1 / (gama - 1))
            # nada[j] = (-((np.linalg.norm(H_v[j].T - (H_v[j].T).dot(S))) ** 2 + a * (
            #         np.linalg.norm(S )) ) / gama) ** (1 / (gama - 1))

    return S

if __name__ == "__main__":
    pass
    # print(X.shape,len(A[0]))

