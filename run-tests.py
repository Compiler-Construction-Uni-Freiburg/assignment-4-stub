import compiler
import interp_Lvar
import interp_Lif
import interp_Cif
import type_check_Lif
import type_check_Cif
from utils import run_tests, enable_tracing

compiler = compiler.Compiler()

enable_tracing()
run_tests("var", compiler, "var", None, interp_Lvar.InterpLvar().interp, type_check_Cif.TypeCheckCif().type_check, interp_Cif.InterpCif().interp)
run_tests("regalloc", compiler, "regalloc", None, interp_Lvar.InterpLvar().interp, type_check_Cif.TypeCheckCif().type_check, interp_Cif.InterpCif().interp)
run_tests("lif", compiler, "lif", type_check_Lif.TypeCheckLif().type_check, interp_Lif.InterpLif().interp, type_check_Cif.TypeCheckCif().type_check, interp_Cif.InterpCif().interp)
