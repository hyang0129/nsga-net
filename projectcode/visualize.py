from collections import namedtuple

import numpy as np
import os
from search import micro_encoding
from graphviz import Digraph
from visualization.micro_visualize import op_labels


def plot_mod(
    genotype_tup: tuple, filename: str = "fn", file_type="pdf", view=True, reduce=False
):
    if reduce:
        genotype = genotype_tup[2]
        concat = genotype_tup[3]
    else:
        genotype = genotype_tup[0]
        concat = genotype_tup[1]
    g = Digraph(
        format=file_type,
        # graph_attr=dict(margin='0.2', nodesep='0.1', ranksep='0.3'),
        edge_attr=dict(fontsize="20", fontname="times"),
        node_attr=dict(
            style="filled",
            shape="rect",
            align="center",
            fontsize="20",
            height="0.5",
            width="0.5",
            penwidth="2",
            fontname="times",
        ),
        engine="dot",
    )
    g.body.extend(["rankdir=LR"])

    g.node("h[i-1]", fillcolor="darkseagreen2")
    g.node("h[i]", fillcolor="darkseagreen2")

    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    print(
        "Rendering %s Cell with Total Add Stages:%i"
        % ("Reduce" if reduce else "Normal", steps)
    )

    for i in range(steps):
        g.node(str(i), label="add %i" % i, fillcolor="lightblue")

    for i in range(steps):
        print("Inputs to Add %i" % i)
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            # g.node(str(steps+k+1), label=op_labels[op], fillcolor='yellow')
            if j == 0:
                u = "h[i-1]"
            elif j == 1:
                u = "h[i]"
            else:
                u = str(j - 2)
            v = str(i)
            # g.edge(u, str(steps+k+1), fillcolor="gray")
            # g.edge(str(steps+k+1), v, fillcolor="gray")
            g.edge(u, v, label=op_labels[op], fillcolor="gray")

            print(" ", op, "from", u)

    g.node("h[i+1]", label="concat", fillcolor="lightpink")

    for i in range(steps):
        if int(i + 2) in concat:
            g.edge(str(i), "h[i+1]", fillcolor="gray")

    print("Final concat of nodes %s" % ",".join([str(i - 2) for i in concat]))

    g.node("output", label="h[i+1]", fillcolor="palegoldenrod")
    g.edge("h[i+1]", "output", fillcolor="gray")

    # g.attr(rank='same')

    g.render(filename, view=view)

    os.remove(filename)
    return g


if __name__ == "__main__":

    # plot an example genmome
    # requires graphviz installed

    g = [
        [
            [np.array([2, 0]), np.array([0, 1])],
            [np.array([5, 0]), np.array([4, 0])],
            [np.array([5, 3]), np.array([2, 2])],
            [np.array([1, 3]), np.array([2, 0])],
            [np.array([5, 0]), np.array([7, 0])],
        ],
        [
            [np.array([6, 0]), np.array([6, 1])],
            [np.array([2, 1]), np.array([5, 1])],
            [np.array([2, 3]), np.array([4, 3])],
            [np.array([6, 1]), np.array([7, 1])],
            [np.array([7, 3]), np.array([7, 4])],
        ],
    ]

    genome = micro_encoding.decode(g)

    print(genome)
    plot_mod(genome)
