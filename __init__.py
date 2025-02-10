import os
os.environ["OMP_NUM_THREADS"] = '6'

import argparse
import numpy as np
from algorithm.FC import fuzzy_c_means, FFCM
from algorithm.experiment import downloadDataset, validation

def main(args):
    """

    :param args:
    :return:
    """

    if args.result == "true":
        Result = []
        Times = 3
        for i in range(Times):
            loader = np.load(
                f"E:\Machine learning\Research\Research-2024.4\A\Code\FFCM1\\new_result\\{args.dataset}-{args.the_number_of_client}-{i}.npz",
                allow_pickle=True)
            L = loader["arr_0"].tolist()
            C = loader["arr_1"].tolist()


            result, msg = validation(L, C, args.dataset)
            Result.append(result)
    else:
        "加载数据集"
        # L_loader = np.load(
        #     f"E:\Machine learning\Research\Research-2024.4\A\Code\Myself4\dataset\{args.dataset}-{args.the_number_of_client}.npz",
        #     allow_pickle=True)
        L_loader = np.load(
            f'E:\Machine learning\Research\Research-2024.4\A\Code\Myself4\\incomplete\{args.dataset}-k-{args.the_number_of_client}.npz',
            allow_pickle=True)
        L = []
        for key in L_loader:
            client = L_loader[key].tolist()
            L.append(client)

        data, label = downloadDataset(args.dataset)
        k = len(np.unique(label))
        # k = 12
        n, d = data.shape
        max_iter = 100
        epsilon = 0.0001
        Times = 5
        m = 2
        Result = []
        i = 0
        while i < Times:
            output = FFCM(L, k, data, m, epsilon, max_iter)
            "保存实验结果"
            # np.savez(
            #     f'E:\Machine learning\Research\Research-2024.4\A\Code\FFCM1\\new_result\\{args.dataset}-{args.the_number_of_client}-{i}.npz',
            #     *output)

            L = output[0]
            C = output[1]
            "实验：性能评估"
            result, msg = validation(L, C, args.dataset)
            if msg == "error":
                Times += 1
                continue
            else:
                i += 1
            Result.append(result)

    # print(f"the number of client:{args.the_number_of_client}")
    print(f"the number of incomplete clusterlet: {args.the_number_of_client}")
    Result = np.stack(Result)
    mean = np.mean(Result, axis=0)
    std = np.std(Result, axis=0)
    print("联邦聚类结果")
    # print(f"purity: {mean[0]:.4f}±{std[0]:.3f}")
    # print(f"ARI: {mean[1]:.4f}±{std[1]:.3f}")
    # print(f"NMI: {mean[2]:.4f}±{std[2]:.3f}")
    # print(f"ACC: {mean[3]:.4f}±{std[3]:.3f}")
    # print(f"FMI: {mean[4]:.4f}±{std[4]:.3f}")
    print(f"purity: {mean[0]:.4f}")
    print(f"ARI: {mean[1]:.4f}")
    print(f"NMI: {mean[2]:.4f}")
    print(f"ACC: {mean[3]:.4f}")
    print(f"FMI: {mean[4]:.4f}")
    # print("kmeans不迭代：")
    # print("kmeans原始数据集的外部指标")
    # print(f"purity: {mean[5]:.4f}±{std[5]:.3f}")
    # print(f"ARI: {mean[6]:.4f}±{std[6]:.3f}")
    # print(f"NMI: {mean[7]:.4f}±{std[7]:.3f}")
    # print(f"ACC: {mean[8]:.4f}±{std[8]:.3f}")
    # print(f"FMI: {mean[9]:.4f}±{std[9]:.3f}")
    # print("kmeans原始数据集的内部指标")
    # print(f"SC: {mean[10]:.4f}±{std[10]:.3f}")
    # print(f"CH: {np.log(mean[11] + 1):.4f}±{np.log(std[11] + 1):.3f}")
    # print(f"DB: {mean[12]:.4f}±{std[12]:.3f}")
    # print("kmeans迭代：")
    # print("kmeans原始数据集的外部指标")
    # print(f"purity: {mean[13]:.4f}±{std[13]:.3f}")
    # print(f"ARI: {mean[14]:.4f}±{std[14]:.3f}")
    # print(f"NMI: {mean[15]:.4f}±{std[15]:.3f}")
    # print(f"ACC: {mean[16]:.4f}±{std[16]:.3f}")
    # print(f"FMI: {mean[17]:.4f}±{std[17]:.3f}")
    # print("kmeans原始数据集的内部指标")
    # print(f"SC: {mean[18]:.4f}±{std[18]:.3f}")
    # print(f"CH: {np.log(mean[19] + 1):.4f}±{np.log(std[19] + 1):.3f}")
    # print(f"DB: {mean[20]:.4f}±{std[20]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo')

    parser.add_argument("--dataset", type=str, help="数据集名称")
    parser.add_argument("--the_number_of_client", type=int, help="client个数")
    parser.add_argument("--result", type=str, help="")

    args = parser.parse_args()

    clients = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # clients = [8]
    # clients = []
    for clients_i in clients:
        args.the_number_of_client = clients_i
        main(args)
        print("-------------------我是分割线------------------")
    # main(args)