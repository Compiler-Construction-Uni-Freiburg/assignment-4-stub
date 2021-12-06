import ast
from ast import *
from utils import *
from x86_ast import *
import os
from typing import Set
from dataclasses import dataclass, field
import platform
from register_allocation import color_graph, build_interference, color_to_register, callee_saved_registers, uncover_live_blocks
from pprint import pprint

Binding = tuple[Name, expr]
Temporaries = list[Binding]

get_fresh_tmp = lambda: generate_name("tmp")

# Tmps and blocks have the same numbering
# -> not an issue, but ugly
def create_block(stmts, basic_blocks):
    label = label_name(generate_name('block'))
    basic_blocks[label] = stmts
    return Goto(label)

@dataclass
class Compiler:
    stack_space: int = 0
    used_callee : list[location] = field(default_factory=list)

    ############################################################################
    # Shrink
    ############################################################################

    def shrink_exp(self, e: expr) -> expr:
        match e:
            # L_if
            # TODO
            # L_var
            case BinOp(left, op, right):
                return BinOp(self.shrink_exp(left), op, self.shrink_exp(right))
            case UnaryOp(op, e):
                return UnaryOp(op, self.shrink_exp(e))
            case _:
                return e

    def shrink_stmt(self, s: stmt) -> stmt:
        match s:
            # L_if
            # TODO
            # L_var
            case Expr(Call(Name('print'), [e])):
                return Expr(Call(Name("print"), [self.shrink_exp(e)]))
            case Expr(e):
                return Expr(self.shrink_exp(e))
            case Assign([Name(var)], e):
                return Assign([Name(var)], self.shrink_exp(e))
            case _:
                return s

    def shrink(self, p: Module) -> Module:
        match p:
            case Module(body):
                return Module([self.shrink_stmt(s) for s in body])

    ############################################################################
    # Remove Complex Operands
    ############################################################################

    def tmps_to_stmts(self, tmps: Temporaries) -> list[stmt]:
        return [Assign([tmp[0]], tmp[1]) for tmp in tmps]

    def rco_exp(self, e: expr, need_atomic: bool) -> tuple[expr, Temporaries]:
        match e:
            # L_if
            # TODO
            # L_var
            case Name(var):
                return (Name(var), [])
            case Constant(_):
                return (e, [])
            case Call(Name('input_int'), []):
                if need_atomic:
                    fresh_tmp = get_fresh_tmp()
                    return (Name(fresh_tmp), [(Name(fresh_tmp), e)])
                return (e, [])
            case UnaryOp(op, e1):
                atm, tmps = self.rco_exp(e1, True)
                if need_atomic:
                    fresh_tmp = get_fresh_tmp()
                    return (Name(fresh_tmp), tmps + [(Name(fresh_tmp), UnaryOp(op, atm))])
                return (UnaryOp(op, atm), tmps)
            case BinOp(e1, op, e2):
                atm1, tmps1 = self.rco_exp(e1, True)
                atm2, tmps2 = self.rco_exp(e2, True)
                if need_atomic:
                    fresh_tmp = get_fresh_tmp()
                    return (Name(fresh_tmp), tmps1 + tmps2 + [(Name(fresh_tmp), BinOp(atm1, op, atm2))])
                return (BinOp(atm1, op, atm2), tmps1 + tmps2)

    def rco_stmt(self, s: stmt) -> list[stmt]:
        match s:
            # L_if
            # TODO
            # L_var
            case Expr(Call(Name('print'), [e])):
                atm, tmps = self.rco_exp(e, True)
                return self.tmps_to_stmts(tmps) + [Expr(Call(Name('print'), [atm]))]
            case Expr(e):
                atm, tmps = self.rco_exp(e, False)
                return self.tmps_to_stmts(tmps) + [Expr(atm)]
            case Assign([Name(var)], e):
                atm, tmps = self.rco_exp(e, False)
                return self.tmps_to_stmts(tmps) + [Assign([Name(var)], atm)]

    def remove_complex_operands(self, p: Module) -> Module:
        match p:
            case Module(body):
                new_body = []
                for s in body:
                    new_body += self.rco_stmt(s)
                return Module(new_body)

    ############################################################################
    # Explicate Control
    ############################################################################

    # Extract side effects from expression statements (ignore results)
    def explicate_effect(self, e, cont, basic_blocks) -> list[stmt]:
        match e:
            case IfExp(test, body, orelse):
                ...
            case Call(func, args):
                ...
            case Let(var, rhs, body):
                ...
            case _:
                ...

    # Generate code for right-hand side of assignment
    def explicate_assign(self, rhs, lhs, cont, basic_blocks) -> list[stmt]:
        match rhs:
            case IfExp(test, body, orelse):
                ...
            case Let(var, rhs, body):
                ...
            case _:
                return [Assign([lhs], rhs)] + cont

    # Generate code for if expression or statement
    def explicate_pred(self, cnd, thn, els, basic_blocks) -> list[stmt]:
        match cnd:
            case Compare(left, [op], [right]):
                goto_thn = create_block(thn, basic_blocks)
                goto_els = create_block(els, basic_blocks)
                return [If(cnd, [goto_thn], [goto_els])]
            case Constant(True):
                return thn
            case Constant(False):
                return els
            case UnaryOp(Not(), operand):
                ...
            case IfExp(test, body, orelse):
                ...
            case Let(var, rhs, body):
                ...
            case _:
                return [If(Compare(cnd, [Eq()], [Constant(False)]),
                           [create_block(els, basic_blocks)],
                           [create_block(thn, basic_blocks)])]

    def explicate_stmt(self, s, cont, basic_blocks) -> list[stmt]:
        match s:
            case Assign([lhs], rhs):
                return self.explicate_assign(rhs, lhs, cont, basic_blocks)
            case Expr(value):
                return self.explicate_effect(value, cont, basic_blocks)
            case If(test, body, orelse):
                ...

    def explicate_control(self, p):
        match p:
            case Module(body):
                new_body = [Return(Constant(0))]
                basic_blocks = {}
                for s in reversed(body):
                    new_body = self.explicate_stmt(s, new_body, basic_blocks)
                basic_blocks[label_name('start')] = new_body
                return CProgram(basic_blocks)

    ############################################################################
    # Select Instructions
    ############################################################################

    def select_arg(self, e: expr) -> arg:
        match e:
            # L_if
            # TODO
            # L_var
            case Constant(n):
                return Immediate(n)
            case Name(var):
                return Variable(var)

    def select_stmt(self, s: stmt) -> list[instr]:
        match s:
            # L_if
            # TODO
            # L_var
            case Assign([Name(var)], BinOp(atm1, Sub(), atm2)):
                arg1 = self.select_arg(atm1)
                arg2 = self.select_arg(atm2)
                match (arg1, arg2):
                    case (Variable(var2), _) if var == var2:
                        return [Instr("subq", [arg2, Variable(var)])]
                    case (_, Variable(var2)) if var == var2:
                        return [Instr("negq", [Variable(var2)]), Instr("addq", [arg1, Variable(var2)])]
                return [Instr("movq", [arg1, Variable(var)]), Instr("subq", [arg2, Variable(var)])]
            case Assign([Name(var)], BinOp(atm1, Add(), atm2)):
                arg1 = self.select_arg(atm1)
                arg2 = self.select_arg(atm2)
                match (arg1, arg2):
                    case (Variable(var2), _) if var == var2:
                        return [Instr("addq", [arg2, Variable(var)])]
                    case (_, Variable(var2)) if var == var2:
                        return [Instr("addq", [arg1, Variable(var)])]
                return [Instr("movq", [arg1, Variable(var)]), Instr("addq", [arg2, Variable(var)])]
            case Assign([Name(var)], UnaryOp(USub(), atm)):
                arg = self.select_arg(atm)
                return [Instr("movq", [arg, Variable(var)]), Instr("negq", [Variable(var)])]
            case Assign([Name(var)], Call(Name("input_int"), [])):
                return [Callq(label_name("read_int"), 0), Instr("movq", [Reg("rax"), Variable(var)])]
            case Assign([Name(var)], atm):
                arg = self.select_arg(atm)
                return [Instr("movq", [arg, Variable(var)])]
            case Expr(Call(Name("print"), [atm])):
                arg = self.select_arg(atm)
                return [Instr("movq", [arg, Reg("rdi")]), Callq("print_int", 1)]
            case Expr(Call(Name("input_int"), [])):
                return [Callq(label_name("read_int"), 0)]
            case Return(atm):
                arg = self.select_arg(atm)
                return [Instr("movq", [arg, Reg("rax")]), Jump("conclusion")]

    def select_instructions(self, p: Module) -> X86Program:
        match p:
            case CProgram(body):
                output = dict()
                output["conclusion"] = [] # will be filled later
                for block in body:
                    new_block = []
                    for stm in body[block]:
                        new_block += self.select_stmt(stm)
                    output[block] = new_block
                return X86Program(output)
        return X86Program([])

    ############################################################################
    # Allocate Registers
    ############################################################################

    def allocate_registers(self, p: X86Program):
        ifg = build_interference(p.body)
        coloring = color_graph(ifg)
        output = dict()
        color_to_location = color_to_register
        offset = 0 
        used_callee = set()

        for key, val in coloring.items():
            if val in color_to_location:
                location = color_to_location[val]
                output[key] = location
                if location in callee_saved_registers:
                    used_callee.add(location)
            else:
                offset -= 8
                color_to_location[key] = Deref("rbp", offset)
                output[key] = color_to_location[key]

        callees_stack_space = len(used_callee) * 8
        for key, loc in output.items():
            match loc:
                case Deref("rbp", offset):
                    output[key] = Deref("rbp", offset - callees_stack_space)

        self.stack_space = -offset
        self.used_callee = list(used_callee)
        return output

    ############################################################################
    # Assign Homes
    ############################################################################

    def assign_homes_arg(self, a: arg, home: dict[Variable, arg]) -> arg:
        match a:
            case Variable(_):
                return home[a]
            case _:
                return a

    def assign_homes_instr(self, i: instr, home: dict[Variable, arg]) -> instr:
        match i:
            case Instr(istr, [arg1, arg2]):
                return Instr(istr, [self.assign_homes_arg(arg1, home),
                                    self.assign_homes_arg(arg2, home)])
            case Instr(istr, [arg1]):
                return Instr(istr, [self.assign_homes_arg(arg1, home)])
            case _:
                return i

    def assign_homes_instrs(
        self, ss: list[instr], home: dict[Variable, arg]
    ) -> list[instr]:
        return [self.assign_homes_instr(istr, home) for istr in ss]

    def assign_homes(self, p: X86Program) -> X86Program:
        home, self.stack_space, self.used_callee = self.allocate_registers(p)
        new_body = dict()
        for block_name, block_items in p.body.items():
            new_body[block_name] = self.assign_homes_instrs(block_items, home)
        return X86Program(new_body)

    ############################################################################
    # Patch Instructions
    ############################################################################

    def patch_instr(self, i: instr) -> list[instr]:
        match i:
            # L_if
            # TODO
            # L_var
            case Instr(istr, [Deref(reg, offset), Deref(reg2, offset2)]):
                if reg == reg2 and offset == offset2:
                    return []
                return [Instr("movq", [Deref(reg, offset), Reg("rax")]),
                        Instr(istr, [Reg("rax"), Deref(reg2, offset2)])]
            case Instr("movq", [arg1, arg2]) if arg1 == arg2:
                    return []
            case _:
                return [i]

    def patch_instrs(self, ss: list[instr]) -> list[instr]:
        output = []
        for s in ss:
            output += self.patch_instr(s)
        return output

    def patch_instructions(self, p: X86Program) -> X86Program:
        new_body = dict()
        for block_name, block_items in p.body.items():
            new_body[block_name] = self.patch_instrs(block_items)
        return X86Program(new_body)

    ############################################################################
    # Prelude & Conclusion
    ############################################################################

    def prelude_and_conclusion(self, p: X86Program) -> X86Program:
        callees_stack_space = len(self.used_callee) * 8
        stack_space_mod16 = self.stack_space if (self.stack_space + callees_stack_space) % 16 == 0 else self.stack_space + 8

        prologue = [Instr("pushq", [Reg("rbp")]), Instr("movq", [Reg("rsp"), Reg("rbp")])]
        for reg in self.used_callee:
            prologue.append(Instr("pushq", [reg]))
        if stack_space_mod16 > 0:
            prologue.append(Instr("subq", [Immediate(stack_space_mod16), Reg("rsp")]))

        p.body["main"] = prologue + [Jump("start")]

        epilogue = [Instr("popq", [Reg("rbp")]), Instr("retq", [])]
        for reg in self.used_callee:
            epilogue.insert(0, Instr("popq", [reg]))
        if stack_space_mod16 > 0:
            epilogue.insert(0, Instr("addq", [Immediate(stack_space_mod16), Reg("rsp")]))

        p.body["conclusion"] = epilogue
        return p



