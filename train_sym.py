
from tqdm import tqdm
import warnings

from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
from sklearn.neighbors import NearestNeighbors
from Adam import adam
from LoadData_MCGC import *
from Metrics_O2MAC import metrics
# import argparse
import yaml
from Initialization import Initail_S
import math



warnings.filterwarnings("ignore")

# parser = argparse.ArgumentParser()
#
# parser.add_argument('--dataset', type=str, default="ACM", help="dataset")




def go_run(dataname, X, gnd):

    
    N = X[0].shape[0]
    I = np.eye(N)

    # attibute matrix

    num_view = len(X) - 1

    print("Now {} are tested! It contains {} views".format(dataname, num_view))
    # Graph filtering
    if "Amazon" in dataname:
        H = X[:2].copy()
        Av = X[3]
        for v in range(num_view):
            k = 1
            A = Av + I
            D = np.sum(A, axis=1)
            D = np.diagflat(D)
            D = np.power(D, -0.5)
            D[np.isinf(D)] = 0
            A = D.dot(A).dot(D)
            Ls = I - A

            while k <= 2:
                for v in range(num_view):
                    H[v] = (I - 0.5 * Ls).dot(H[v])
                k += 1

    else:
        H = []
        for v in range(num_view):
            H.append(X[0])

        Av = X[1]
        for A_ in X[1:]:
            k = 1
            A = A_ + I
            D = np.sum(A, axis=1)
            D = np.diagflat(D)
            D = np.power(D, -0.5)
            D[np.isinf(D)] = 0
            A = D.dot(A).dot(D)
            Ls = I - A

            while k <= 2:
                for v in range(num_view):
                    H[v] = (I - 0.5 * Ls).dot(H[v])
                k += 1



    kkk = len(np.unique(gnd))


    list_a = [0.001, 1, 10, 100, 1000]
    gamas = [-1, -2, -3, -4, -5]
    nada = [1 / num_view for i in range(num_view)]




    epochs = 100
    do_times = 3

    H_Ht = []
    for v in range(num_view):
        H_Ht.append(H[v].dot(H[v].T))

    print('Begin!\n')



    #Getting NBrs 
    nbrs_inx = []
    try:
        for v in range(num_view):
            idx = np.load("./nbrs10_{}_view{}.npy".format(dataname, v))
            idx = idx.astype(np.int)
            nbrs_inx.append(idx.astype(int))

    except Exception:
        for v in range(num_view):
            X_nb = np.array(H[v])
            nbrs_v = np.zeros((N, 10))
            nbrs = NearestNeighbors(n_neighbors=11, algorithm='auto').fit(X_nb)
            dis, idx = nbrs.kneighbors(X_nb)
            for i in range(N):
                for j in range(10):
                    nbrs_v[i][j] += idx[i][j + 1]
            # svaing for cheap computing
            np.save("./nbrs10_{}_view{}.npy".format(dataname, v), nbrs_v)
            nbrs_inx.append(np.array(nbrs_v).astype(int))

    best_S = np.zeros((N, N))
    for a in list_a:
        for gama in gamas:
            re = []
            nmi_epoch = 0
            ari_epoch = 0
            f1_epoch = 0
            best_epoch = 0

            #Initial
            f = open('config.yaml')
            config_data = yaml.load(f)
            initial_a = config_data['{}'.format(dataname)]['alpha']
            initial_gama = config_data['{}'.format(dataname)]['gama']

            if dataname == "ACM":
                S_re = np.load('./Initialization/ACM_initialS.npy'.format(dataname))
            else:
                S_re = Initail_S(H_v= H, av=Av, a=initial_a, gama=initial_gama)
            # S_re = 0.5 * (np.fabs(S_) + np.fabs(S_.T))
            best_S = best_S + S_re
            print('Begin S\n')

            acc_epoch = 0
            for do_tims in range(do_times):
                # checkPoint_1
                trigger = 0
                cut_point = 0


                # 梯度更新
                cf = None
                loss_last = 1e16

                for m in range(N):
                    S_re[m][m] = 0
                for epoch in range(epochs):
                    # 取最近邻
                    if trigger >= 10:
                        break

                    grad = np.zeros((N, N))
                    H_Ht_S = []
                    # S = S_re.copy()
                    for v in range(num_view):
                        H_Ht_S.append(H_Ht[v].dot(S_re))
                    # 梯度更新

                    for i in tqdm(range(N)):
                        k0 = np.exp(S_re[i]).sum() - np.exp(S_re[i][i])
                        # H = S[:, i].sum() - 1
                        for j in range(i, N):
                            F11 = 0
                            F12 = 0
                            F2 = 0
                            for v in range(num_view):
                                F11 = F11 + nada[v] * H_Ht[v][i][j]
                                F12 = F12 + nada[v] * H_Ht_S[v][i][j]
                                if i != j:
                                    if j in nbrs_inx[v][i]:
                                        F2 = F2 + nada[v] * (-1 + 10 * np.exp(S_re[i][j]) / k0)
                                    else:
                                        F2 = F2 + nada[v] * (10 * np.exp(S_re[i][j]) / k0)

                            F1 = -2 * F11 + 2 * F12
                            grad[i][j] = a * F2 + F1
                            grad[j][i] = grad[i][j]

                    loss_all_node = 0
                    loss_view = 0
                    for v in range(num_view):
                        for i in (range(N)):
                            k0 = np.exp(S_re[i]).sum() - np.exp(S_re[i][i])
                            loss_nbr = 0
                            for z in range(10):
                                if nbrs_inx[v][i][z] != i:
                                    loss_nbr = loss_nbr - np.log(np.exp(S_re[i][nbrs_inx[v][i][z]]) / k0)
                            loss_all_node = loss_all_node + loss_nbr
                        loss_view = loss_view + np.linalg.norm(
                            H[v].T - H[v].T.dot(S_re)) ** 2

                    loss_S_re = a * loss_all_node + loss_view

                    # checkPoint_1
                    oder = math.log10(loss_S_re)
                    oder = int(oder)
                    oder = min(oder, 3)
                    Tol = 1 * math.pow(10, -oder)
                    # print("Now Tol======>{}".format(Tol))
                    if math.fabs(loss_S_re - loss_last) <= math.fabs(Tol * loss_S_re):
                        break
                    else:
                        loss_last = loss_S_re

                    if loss_S_re > loss_last:
                        cut_point += 1
                    if cut_point > 2:
                        if trigger >= 5:
                            break

                    S_re, cf = adam(S_re, grad, cf)

                    #Clustering

                    C = 0.5 * (np.fabs(S_re) + np.fabs(S_re.T))
                    u, s, v = sp.linalg.svds(C, k=kkk, which='LM')
                    # 聚类
                    kmeans = KMeans(n_clusters=kkk, random_state=23).fit(u)
                    predict_labels = kmeans.predict(u)
                    # predict_labels = SpectralClustering(n_clusters=kkk, gamma=-4).fit_predict(u)


                    # 几个metric
                    re_ = metrics.clustering_metrics(gnd, predict_labels)
                    ac, nm, ari, f1 = re_.evaluationClusterModelFromLabel(gama, kkk, a)
                    print("Now ACC ==>{}".format(ac))
                    if ac > acc_epoch:
                        acc_epoch = ac
                        best_S = S_re
                        nmi_epoch = nm
                        ari_epoch = ari
                        f1_epoch = f1
                        best_epoch = do_tims * epochs + epoch
                    else:
                        trigger += 1
                    print(acc_epoch)

                    # loss = np.linalg.norm((X.T - X.T.dot(S_re))) ** 2 + loss_all_node
                    # num_los = loss
                    # print('epoch {} time, loss is {} \n'.format(epoch, loss))

                    # print('num{} time, loss is {} \n'.format(epoch, loss))

                    # 更新lambda
                for v in range(num_view):
                    loss_all_node = 0
                    for i in tqdm(range(N)):
                        k0 = np.exp(S_re[i]).sum() - np.exp(S_re[i][i])
                        loss_nbr = 0
                        for z in range(10):
                            if nbrs_inx[v][i][z] != i:
                                loss_nbr = loss_nbr - np.log(np.exp(S_re[i][nbrs_inx[v][i][z]]) / k0)
                        loss_all_node = loss_all_node + loss_nbr

                    nada[v] = (-((np.linalg.norm(
                        H[v].T - H[v].T.dot(best_S)) ** 2 + a * loss_all_node) / gama)) ** (1 / (gama - 1))
                    # nada[j] = (-((np.linalg.norm(X_bar[j].T - (X_bar[j].T).dot(S)))**2 + a * (np.linalg.norm(S - A_[j])) ** 2 ) / gama) ** (1 / (gama - 1))
                    print("nada{}值".format(v))
                    print(nada[v])

            re.append(acc_epoch)
            re.append(nmi_epoch)
            re.append(ari_epoch)
            re.append(f1_epoch)
            re.append(best_epoch)

            # np.savetxt('Mv_ACM_nbrs_{}a_{}bestU.txt'.format(num_nbr, a),best_S, delimiter=',')
            # np.savetxt('{}_nbrs10_a{}_gama{}_bestRe_sym_test1.txt'.format(dataname, a, gama), re, delimiter=',')
            print(re)
    # np.savetxt("{}_nbrs10_bestS.txt".format(dataname), best_S, delimiter=',')
    # np.savetxt('SingleView_nbrs{}_a{}__epoch{}_S.txt'.format(num_nbr,a, epoch), S_re, delimiter=',')
    # print("Saved!")





if __name__ == "__main__":
    Switcher = {
            0: Acm,
            1: Dblp,
            2: Imdb,
            3: mine_Amazon_normolized,
            4: mine_Amazon_normolized_com,

    }
    for i, dataname in enumerate(["ACM", "DBLP", "IMDB", "Amazon photos", "Amazon Computers"]):
        X, gnd = Switcher[i]()
        go_run(X=X, gnd=gnd, dataname=dataname)
