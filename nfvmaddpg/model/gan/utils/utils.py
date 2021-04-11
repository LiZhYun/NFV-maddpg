import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

if __name__ == '__main__':
    from vnf_metrics import VNFMetrics
else:
    from nfvmaddpg.model.gan.utils.vnf_metrics import VNFMetrics


def topos2grid_image(topos, vnf_decoder):
    # topos = [e if e is not None else Chem.RWMol() for e in topos]
    subset_color = [
        "antiquewhite",
        "blanchedalmond",
        "coral",
        "dodgerblue",
        "firebrick",
        "gainsboro",
        "honeydew",
        "indigo",
        "khaki",
        "limegreen",
        "mistyrose",
        "navy",
        "orchid",
        "plum",
        "rosybrown",
        "silver",
        "teal"
    ]
    
    for num, topo in enumerate(topos):
        if topo is None:
            continue
        # nx.draw(topo, with_labels=True, font_weight='bold')
        labels = dict()
        for node in list(topo.nodes):
            labels[node] = [vnf_decoder[index]
                            for index, usage in enumerate(topo.nodes[node]['usagelist']) if usage != 0]
            if labels[node] == []:
                labels[node] = ''
            else:
                labels[node] = ' '.join(labels[node])
        elarge = [(u, v)
                  for (u, v, d) in topo.edges(data=True) if d["edgeDatarate"] > 0.5]
        esmall = [(u, v) for (u, v, d) in topo.edges(
            data=True) if d["edgeDatarate"] <= 0.5]
        pos = nx.spring_layout(topo)  # positions for all nodes
        # nodes
        colorindex = np.random.randint(17)
        color = [subset_color[colorindex]
                 for v, data in topo.nodes(data=True)]
        # node_size = 0.5 + np.random.random(len(topo.nodes)) / 2 * 1000
        nx.draw_networkx_nodes(
            topo, pos, node_color=color, node_size=500)
        # edges
        nx.draw_networkx_edges(topo, pos, edgelist=elarge,
                               width=3, edge_color="black")
        nx.draw_networkx_edges(
            topo, pos, edgelist=esmall, width=1.5, alpha=0.5, edge_color="black", style="dashed"
        )
        # labels
        nx.draw_networkx_labels(topo, pos, labels=labels, font_size=10,
                                font_family="sans-serif")
        # nx.draw_networkx_labels(topo, pos=pos, labels=labels)
        plt.axis("off")
        plt.savefig('pics/topo_' + str(num) + '.jpg')
        plt.show()
        
    # return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(150, 150))


# def classification_report(data, model, session, sample=False):
#     _, _, _, a, x, _, f, _, _ = data.next_validation_batch()

#     n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
#         model.nodes_argmax, model.edges_argmax], feed_dict={model.edges_labels: a, model.nodes_labels: x,
#                                                             model.node_features: f, model.training: False,
#                                                             model.variational: False})
#     n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

#     y_true = e.flatten()
#     y_pred = a.flatten()
#     target_names = [str(Chem.rdchem.BondType.values[int(e)]) for e in data.bond_decoder_m.values()]

#     print('######## Classification Report ########\n')
#     print(sk_classification_report(y_true, y_pred, labels=list(range(len(target_names))),
#                                    target_names=target_names))

#     print('######## Confusion Matrix ########\n')
#     print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))

#     y_true = n.flatten()
#     y_pred = x.flatten()
#     target_names = [Chem.Atom(e).GetSymbol() for e in data.atom_decoder_m.values()]

#     print('######## Classification Report ########\n')
#     print(sk_classification_report(y_true, y_pred, labels=list(range(len(target_names))),
#                                    target_names=target_names))

#     print('\n######## Confusion Matrix ########\n')
#     print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))


# def reconstructions(data, model, session, batch_dim=10, sample=False):
#     m0, _, _, a, x, _, f, _, _ = data.next_train_batch(batch_dim)

#     n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
#         model.nodes_argmax, model.edges_argmax], feed_dict={model.edges_labels: a, model.nodes_labels: x,
#                                                             model.node_features: f, model.training: False,
#                                                             model.variational: False})
#     n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

#     m1 = np.array([e if e is not None else Chem.RWMol() for e in [data.matrices2mol(n_, e_, strict=True)
#                                                                   for n_, e_ in zip(n, e)]])

#     mols = np.vstack((m0, m1)).T.flatten()

#     return mols


def samples(data, model, session, embeddings, sample=False):
    n, e = session.run([model.nodes_gumbel_softmax, model.edges_logits] if sample else [
        model.nodes_gumbel_softmax, model.edges_logits], feed_dict={
        model.embeddings: embeddings, model.training: False})
    # n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    n, e = standardization(n), standardization(e)

    mols = [data.matrices2topo(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    return mols


def all_scores(topos, data, norm=False, reconstruction=False):
    m0 = {k: list(filter(lambda e: e is not None, v)) for k, v in {
        'typenum score': VNFMetrics.vnftype_num(topos, data.vnf_decoder)}.items()}

    # m1 = {'valid score': MolecularMetrics.valid_total_score(mols) * 100,
    #       'unique score': MolecularMetrics.unique_total_score(mols) * 100,
    #       'novel score': MolecularMetrics.novel_total_score(mols, data) * 100}

    return m0

def standardization(data):
    mu = np.min(data, axis=-1)
    sigma = np.max(data, axis=-1)
    return (data - mu[:, :, np.newaxis]) / (sigma[:, :, np.newaxis] - mu[:, :, np.newaxis] + 3.14e-8)

def standardization2(data):
    mu = np.min(data, axis=-1)
    sigma = np.max(data, axis=-1)
    return (data - mu[:, np.newaxis]) / (sigma[:, np.newaxis] - mu[:, np.newaxis] + 3.14e-8)


