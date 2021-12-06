from x86_ast import *
from typing import Callable
from graph import *
from priority_queue import *
from pprint import pprint

caller_saved_registers: set[location] = set(
    [
        Reg("rax"),
        Reg("rcx"),
        Reg("rdx"),
        Reg("rsi"),
        Reg("rdi"),
        Reg("r8"),
        Reg("r9"),
        Reg("r10"),
        Reg("r11"),
    ]
)
callee_saved_registers: set[location] = set(
    [Reg("rsp"), Reg("rbp"), Reg("rbx"), Reg("r12"), Reg("r13"), Reg("r14"), Reg("r15")]
)
registers_for_coloring = [
    Reg("rbp"),
    Reg("rsp"),
    Reg("rax"),
    Reg("rcx"),
    Reg("rdx"),
    Reg("rsi"),
    Reg("rdi"),
    Reg("r8"),
    Reg("r9"),
    Reg("r10"),
    Reg("r11"),
    Reg("rbx"),
    Reg("r12"),
    Reg("r13"),
    Reg("r14"),
    Reg("r15"),
]
# Mapping from registers to colors
register_to_color = {
    registers_for_coloring[i]: i - 3 for i in range(0, len(registers_for_coloring))
}
# The inverse of the above
color_to_register = {v: k for k, v in register_to_color.items()}

argument_passing_registers = lambda arity: set([Reg("rdi"), Reg("rsi"), Reg("rdx"), Reg("rcx"), Reg("r8"), Reg("r9")][:arity])


def get_arg_locations(arg1 : arg) -> set[location]:
    match arg1:
        # L_if
        # TODO
        # L_var
        case Reg(_):
            return set([arg1])
        case Variable(_):
            return set([arg1])
        case _:
            return set()

def get_read_write_locations(istr : Instr) -> tuple[set[location], set[location]]:
    match istr:
        # L_if
        # TODO
        # L_var
        case Instr(op, [arg1, arg2]):
            if op == "addq" or op == "subq":
                return (get_arg_locations(arg1) | get_arg_locations(arg2), get_arg_locations(arg2))
            else:
                return (get_arg_locations(arg1), get_arg_locations(arg2))
        case Instr("negq", [arg]):
            locs = get_arg_locations(arg)
            return (locs, locs)
        case Instr("pushq", [arg]):
            return (get_arg_locations(arg), set())
        case Instr("popq", [arg]):
            return (set(), get_arg_locations(arg))
        case Callq(l, i):
            return (argument_passing_registers(i), caller_saved_registers)
        case _:
            return (set(), set())

def cfg(basic_blocks: dict[str, list[instr]]) -> DirectedAdjList:
    g = DirectedAdjList()
    # TODO
    return g
    
def get_liveness_order(basic_blocks: dict[str, list[instr]]) -> list:
    g = transpose(cfg(basic_blocks))
    return topological_sort(g)

def uncover_live(order, basic_blocks: dict[str, list[instr]]) \
        -> dict[str, list[set[location]]]: 
    live_before_block: dict[str, set[location]] = dict()
    output: dict[str, list[set[location]]] = dict()
    for block_name in order:
        #TODO
        ...
    return output

def build_interference(basic_blocks: dict[str, list[instr]]) -> UndirectedAdjList:
    graph = UndirectedAdjList()
    order = get_liveness_order(basic_blocks)
    liveblocks = uncover_live(order, basic_blocks)
    for block_name in order:
        istrs = basic_blocks[block_name]
        live = liveblocks[block_name]
        for s in live:
            for v in s:
                graph.add_vertex(v)
        for i in range(len(istrs)):
            istr = istrs[i]
            L_after = live[i + 1]
            match istr:
                # L_if
                # TODO
                # L_var
                case Instr("movq", [s, d]):
                    for v in L_after:
                        if v != s and v != d and not graph.has_edge(v, d):
                            graph.add_edge(v, d)
                case _:
                    _, W = get_read_write_locations(istr)
                    for v in L_after:
                        for d in W:
                            if v != d and not graph.has_edge(v, d):
                                graph.add_edge(v, d)
    return graph

def pre_color(graph: UndirectedAdjList) -> tuple[dict[location, int], dict[location, set[int]]]:
    coloring: dict[location, int] = dict()
    saturations: dict[location, set[int]] = dict()
    vertices: list[location] = list(graph.vertices())
    for v in vertices:
        saturations[v] = set()
    for v in vertices:
        match v:
            case Reg(r):
                coloring[v] = register_to_color[v]
                for u in set(graph.adjacent(v)):
                    saturations[u].add(register_to_color[v])
            case _:
                continue
    return coloring, saturations

def color_graph(graph : UndirectedAdjList) -> dict[Variable, int]:
    coloring, saturations = pre_color(graph)
    vertices = list(graph.vertices())
    pq = PriorityQueue(lambda x, y: len(saturations[x.key]) < len(saturations[y.key]))

    for v in vertices:
        pq.push(v)
    while not pq.empty():
        v = pq.pop()
        number = 0
        while number in saturations[v]:
            number += 1
        coloring[v] = number
        for u in set(graph.adjacent(v)):
            saturations[u].add(number)
            pq.increase_key(u)

    # Filter out registers for output
    output: dict[Variable, int] = dict()
    for key, val in coloring.items():
        match key:
            case Variable(v):
                output[key] = val
            case _:
                continue
    return output
