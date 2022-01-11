import networkx as nx
from pygsp import graphs, filters
from matplotlib import pyplot as plt
from networkx.linalg import laplacianmatrix
from scipy.special import softmax
import numpy as np
import cvxopt
# from cvxopt import matrix
# from cvxopt import solvers
import cvxpy as cvx
import cvxpy.utilities
from cvxpy import atoms
from qcqp import QCQP
from qcqp import SDR
from qcqp import *
import logging
import mosek


g_raw_data_path = 'EdgeList.ORIGINALCAN.THRBIAS.csv'
g_digraph_gml_path = 'thrbias.gml'
g_sol_path = 'solution.txt'


def raw_to_gml():
    thrbias_graph = nx.DiGraph()
    with open(g_raw_data_path, 'r') as in_fd:
        csv_line = in_fd.readline()
        csv_line = in_fd.readline()
        while csv_line:
            l_fields = csv_line.split(',')
            src_node = l_fields[0].strip()
            trg_node = l_fields[1].strip()
            edge_weight = float(l_fields[2].strip())
            thrbias_graph.add_edge(src_node, trg_node, weight=edge_weight)
            csv_line = in_fd.readline()
        in_fd.close()
    nx.write_gml(thrbias_graph, g_digraph_gml_path)
    print(nx.info(thrbias_graph))
    nx.draw(thrbias_graph)
    plt.show()


def load_digraph_from_gml():
    nx_digraph = nx.read_gml(g_digraph_gml_path)
    adj_mat = nx.adjacency_matrix(nx_digraph)
    adj_mat = adj_mat.todense()
    pos_adj_mat = softmax(adj_mat)
    gsp_adj_mat = (pos_adj_mat + pos_adj_mat.T) / 2
    gsp_graph = graphs.Graph(gsp_adj_mat)
    return gsp_graph, nx_digraph


def compute_fourier(gsp_digraph, nx_digraph=None):
    # di_lap = laplacianmatrix.directed_laplacian_matrix(nx_digraph)
    # gsp_digraph.L = di_lap
    gsp_digraph.compute_laplacian(lap_type='normalized')
    gsp_digraph.compute_fourier_basis()
    # print()


def build_init_graph_signal(l_nodes, idx):
    ret_signal = np.zeros(len(l_nodes))
    ret_signal[idx] = 1.0
    return ret_signal


def signal_filtering(a_init_graph_signal, gsp_graph):
    heat_filter = filters.Heat(gsp_graph, tau=[1, 5, 10], normalize=False)
    filtered_signal = heat_filter.filter(a_init_graph_signal, method='exact')
    ret_signal = filtered_signal[:, 0] + filtered_signal[:, 1] + filtered_signal[:, 2]
    ret_signal = softmax(ret_signal)
    return ret_signal


def sythesized_signal_filtering(gsp_graph, l_nodes):
    ret_signal = np.zeros(len(l_nodes))
    for i in range(0, len(l_nodes)):
        init_signal = build_init_graph_signal(l_nodes, i)
        filtered_signal = signal_filtering(init_signal, gsp_graph)
        ret_signal += filtered_signal
    ret_signal = softmax(ret_signal)
    return ret_signal


def bias_signal_filtering(gsp_graph, l_nodes):
    idx = l_nodes.index('Bias')
    init_signal = build_init_graph_signal(l_nodes, idx)
    return signal_filtering(init_signal, gsp_graph)


def compute_loss(nx_digraph, graph_signal):
    l_nodes = list(nx_digraph.nodes)
    tv = 0.0
    for edge in nx_digraph.edges.data('weight'):
        src = edge[0]
        trg = edge[1]
        weight = edge[2]


# def quad_sol_by_cvxopt(nx_digraph):
#     l_nodes = [node for node in list(nx_digraph.nodes) if node != 'Bias']
#     nx_sub_digraph = nx.subgraph(nx_digraph, l_nodes)
#     sub_digraph_adj_mat = nx.adjacency_matrix(nx_sub_digraph)
#
#     a_bias_out = np.zeros(len(l_nodes))
#     a_bias_in = np.zeros(len(l_nodes))
#     for edge in nx_digraph.edges.data('weight'):
#         src = edge[0]
#         trg = edge[1]
#         weight = edge[2]
#         if src == 'Bias':
#             idx = l_nodes.index(trg)
#             a_bias_out[idx] = weight
#             continue
#         if trg == 'Bias':
#             idx = l_nodes.index(src)
#             a_bias_in[idx] = weight
#             continue
#     sym_a_bias = 0.5 * (a_bias_out + a_bias_in)
#
#     x_low_bound_mat = np.zeros((len(l_nodes), len(l_nodes)))
#     np.fill_diagonal(x_low_bound_mat, -1.0)
#     x_up_bound_mat = np.zeros((len(l_nodes), len(l_nodes)))
#     np.fill_diagonal(x_up_bound_mat, 1.0)
#
#     x_low_bound_arr = np.zeros(len(l_nodes))
#     x_up_bound_arr = np.full(len(l_nodes), 1.0)
#
#     max_adj_mat = np.full(sub_digraph_adj_mat.shape, np.max(sub_digraph_adj_mat))
#     conj_adj_mat = max_adj_mat - sub_digraph_adj_mat
#     sym_adj_mat = conj_adj_mat + conj_adj_mat.T
#     # sym_adj_mat = sym_adj_mat.todense()
#     P = matrix(sym_adj_mat, tc='d')
#     q = matrix(sym_a_bias, tc='d')
#     G = matrix(np.append(x_low_bound_mat, x_up_bound_mat, axis=0), tc='d')
#     h = matrix(np.append(x_low_bound_arr, x_up_bound_arr), tc='d')
#     # A = matrix(0.0, (1, len(l_nodes)))
#     # b = matrix(0.0)
#
#     sol = solvers.qp(P, q, G, h, solver='mosek')
#     print()


def quad_sol_by_qcqp(nx_digraph):
    l_nodes = [node for node in list(nx_digraph.nodes) if node != 'Bias']
    nx_sub_digraph = nx.subgraph(nx_digraph, l_nodes)
    sub_digraph_adj_mat = nx.adjacency_matrix(nx_sub_digraph)
    x_dim = len(l_nodes)

    a_bias_out = np.zeros(len(l_nodes))
    a_bias_in = np.zeros(len(l_nodes))
    for edge in nx_digraph.edges.data('weight'):
        src = edge[0]
        trg = edge[1]
        weight = edge[2]
        if src == 'Bias':
            idx = l_nodes.index(trg)
            a_bias_out[idx] = weight
            continue
        if trg == 'Bias':
            idx = l_nodes.index(src)
            a_bias_in[idx] = weight
            continue
    sym_a_bias = 0.5 * (a_bias_out + a_bias_in)

    x_low_bound_mat = np.zeros((len(l_nodes), len(l_nodes)))
    # np.fill_diagonal(x_low_bound_mat, -1.0)
    np.fill_diagonal(x_low_bound_mat, 1.0)
    x_up_bound_mat = np.zeros((len(l_nodes), len(l_nodes)))
    np.fill_diagonal(x_up_bound_mat, 1.0)

    x_low_bound_arr = np.zeros(len(l_nodes))
    x_up_bound_arr = np.full(len(l_nodes), 1.0)

    max_adj_mat = np.full(sub_digraph_adj_mat.shape, np.max(sub_digraph_adj_mat))
    conj_adj_mat = max_adj_mat - sub_digraph_adj_mat
    # sym_adj_mat = conj_adj_mat + conj_adj_mat.T
    sym_adj_mat = 0.5 * (sub_digraph_adj_mat + sub_digraph_adj_mat.T)
    sym_adj_mat = sym_adj_mat.todense()

    P = sym_adj_mat
    q = sym_a_bias
    G = np.append(x_low_bound_mat, x_up_bound_mat, axis=0)
    h = np.append(x_low_bound_arr, x_up_bound_arr)
    # P = matrix(sym_adj_mat, tc='d')
    # q = matrix(sym_a_bias, tc='d')
    # G = matrix(np.append(x_low_bound_mat, x_up_bound_mat, axis=0), tc='d')
    # h = matrix(np.append(x_low_bound_arr, x_up_bound_arr), tc='d')

    x = cvx.Variable(x_dim)
    obj = atoms.quad_form(x, P) + q * x
    # cons = [G * x <= h]
    # cons = [x_low_bound_mat * x >= x_low_bound_arr, x_up_bound_mat * x <= x_up_bound_arr]
    # cons = [x_low_bound_mat * x >= x_low_bound_arr]
    cons = [cvx.square(x) <= 1, x_low_bound_mat * x >= x_low_bound_arr]
    # prob = cvx.Problem(cvx.Minimize(obj), cons)
    prob = cvx.Problem(cvx.Maximize(obj), cons)

    qcqp = QCQP(prob)
    qcqp.suggest(RANDOM, solver=cvx.MOSEK)
    # print("SDR lower bound: %.3f" % qcqp.sdr_bound)


    # f_cd, v_cd = qcqp.improve(ADMM)
    # print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))
    final_ret = (atoms.quad_form(x, P) + q * x).value
    print('objective = %s' % final_ret)
    print(x.value)

    output_node_list(l_nodes)


def output_node_list(l_nodes):
    # l_nodes = [node for node in list(nx_digraph.nodes) if node != 'Bias']
    with open(g_sol_path, 'w+') as out_fd:
        for idx, node in enumerate(l_nodes):
            out_fd.write(node)
            out_fd.write('\n')
        out_fd.close()


def main():
    # raw_to_gml()
    gsp_graph, nx_digraph = load_digraph_from_gml()
    # compute_fourier(gsp_graph, nx_digraph)
    # a_init_graph_signal = build_init_graph_signal(list(nx_digraph.nodes))
    # a_filtered_graph_signal = bias_signal_filtering(gsp_graph, list(nx_digraph.nodes))
    quad_sol_by_qcqp(nx_digraph)
    print()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()