# objects here should be moved to a proper package

from search import micro_encoding
import pymoo
import torch
import os


def decode_individual(individual: pymoo.model.individual.Individual) -> tuple:
    """
    Takes an individual, with a genome as a numpy array, and decodes it into a
    human friendly genome.

    Args:
        individual:

    Returns:

    """

    return micro_encoding.decode(micro_encoding.convert(individual.X))


def convert_mutation_map(individual: pymoo.model.individual.Individual) -> tuple:
    """
    Converts a mutation map that describes what genes mutated into a format that matches
    the genome length.

    Args:
        individual:

    Returns:
        normal_cell_map, reduce_cell_map
    """
    genome_length = len(decode_individual(individual).normal)

    normal_muta_map = np.all(
        individual.mutation_map[: genome_length * 2].reshape(genome_length, 2), axis=1
    )
    reduce_muta_map = np.all(
        individual.mutation_map[genome_length * 2 :].reshape(genome_length, 2), axis=1
    )

    return normal_muta_map, reduce_muta_map


def read_parent_by_id(id: int, expr_root: str) -> dict:
    """
    based on the id, find the directory of the parent and parse its relevant data for wt inheritance.

    Args:
        id: one of the parent ids saved on the individual after traced mating
        expr_root: this should match args.save_path

    Returns:
        A dictionary describing the parent
    """
    # parent is a dict
    arch_path = os.path.join(expr_root, "arch_{}".format(id))

    path = os.path.join(arch_path, "log.txt")
    wt_path = os.path.join(arch_path, "weights.pt")

    with open(path, "r") as f:
        s = f.read()

    genome = eval(s.split("\n")[1][15:])
    flops = float(s.split("\n")[3][8:-2])
    acc = float(s.split("\n")[4][11:])
    error = 100 - acc
    weights = torch.load(wt_path)
    parent = {"genome": genome, "flops": flops, "error": error, "weights": weights}
    return parent
