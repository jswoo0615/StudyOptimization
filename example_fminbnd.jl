include("julia_fminbnd.jl")
# --- 6. fminbnd 예제 ---
println("\n--- Running fminbnd equivalent example ---")
# Minimize x^2 - 4x + 4 subject to 0 <= x <= 3
f_bnd = (x) -> x^2 - 4*x + 4 # 단일 변수 Julia 함수
x1_bnd = 0.0 # 하한
x2_bnd = 3.0 # 상한
bnd_result = julia_fminbnd(f_bnd, x1_bnd, x2_bnd, iterations=1000, g_tol=1e-5)
println("Result: ", bnd_result)