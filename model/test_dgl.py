import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.softmax import edge_softmax

'''
0 -> r0 -> 1
0 -> r2 -> 4
3 -> r2 -> 2
4 -> r1 -> 1
'''


def create_graph():
    g = dgl.DGLGraph()
    g.add_nodes(5)
    g.add_edges([1, 4, 2, 1], [0, 0, 3, 4])
    g.readonly()
    g.ndata['id'] = torch.arange(5, dtype=torch.long)
    g.edata['type'] = torch.LongTensor([0, 2, 2, 1])
    return g
    

def compute_attention(g, W_R, n_relations):

    def att_score(edges):
        # formula (4)
        r_mul_t = torch.matmul(entity_embedding[edges.src['id']], W_r)                                  # (n_edge, relation_dim)
        r_mul_h = torch.matmul(entity_embedding[edges.dst['id']], W_r)                                  # (n_edge, relation_dim)
        r_embed = entity_embedding[edges.data['type']]                                                  # (1, relation_dim)
        att = torch.bmm(r_mul_t.unsqueeze(1), torch.tanh(r_mul_h + r_embed).unsqueeze(2)).squeeze(-1)   # (n_edge, 1)
        return {'att': att}

    g = g.local_var()
    for i in range(n_relations):
        edge_idxs = g.filter_edges(lambda edge: edge.data['type'] == i)
        W_r = W_R[i]
        g.apply_edges(att_score, edge_idxs)

    # formula (5)
    g.edata['att'] = edge_softmax(g, g.edata.pop('att'))
    return g.edata.pop('att')


def gnn(g):
    g = g.local_var()
    g.ndata['node'] = entity_embedding
    g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), dgl.function.sum('side', 'N_h'))
    g.ndata['sum'] = g.ndata['N_h'] + g.ndata['node']
    print(g.ndata['N_h'])
    print(g.ndata['node'])
    print(g.ndata['sum'])



if __name__ == '__main__':
    relation_embedding = nn.Parameter(torch.FloatTensor([[1, 2], [3, 4], [5, 6]]))
    entity_embedding = nn.Parameter(torch.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))
    W_R = nn.Parameter(torch.arange(1, 13).reshape([3, 2, 2]).float())
    print(relation_embedding)
    print(entity_embedding)
    print(W_R)

    g = create_graph()

    att = compute_attention(g, W_R, 3)
    g.edata['att'] = att
    print(att)

    gnn(g)



