# julia_lsqcurvefit.jl 파일을 포함합니다.
# 이 파일에는 julia_lsqcurvefit 함수 정의가 포함되어 있어야 합니다.
include("julia_lsqcurvefit.jl")

# --- julia_lsqcurvefit 사용 예제 ---
# 이 예제는 Wrapper 함수 외부에서 실행되어야 합니다.

using LsqFit # Wrapper 함수를 include 하므로 LsqFit도 필요
using Plots # 시각화를 위해 추가
using Random # 데이터 생성을 위해 추가

# 예제 데이터 생성 및 모델 함수 정의
# 모델 함수: y = p[1] * exp(-p[2] * t)
model_exp(t, p) = p[1] .* exp.(-p[2] .* t)

# 실제 파라미터 값
p_true_exp = [10.0, 0.5]
# 독립 변수 데이터 (시간 t)
t_data_exp = range(0.5, stop=10.0, length=20)
# 모델 출력에 노이즈 추가하여 실제 측정 데이터처럼 만듭니다.
Random.seed!(1234); # 결과 재현성을 위해 시드 고정 (선택 사항)
y_data_exp = model_exp(t_data_exp, p_true_exp) .+ 0.5 .* randn(length(t_data_exp))

# 파라미터 초기 추정값
p_init_exp = [8.0, 0.8]

# --- julia_lsqcurvefit 함수 호출 및 결과 출력 (경계 조건 없음) ---
println("\n--- julia_lsqcurvefit 실행 결과 (경계 조건 없음) ---")
result_curvefit = julia_lsqcurvefit(model_exp, t_data_exp, y_data_exp, p_init_exp)

# 결과 출력 부분을 보기 좋게 수정
# LsqFit.converged(result.lsqfit_result) 대신 result.lsqfit_result.converged 필드 사용
println("  최적화 상태: ", result_curvefit.lsqfit_result.converged ? "수렴됨" : "수렴되지 않음")
println("  최적화된 파라미터 (solution): ", result_curvefit.solution)
println("  최소 잔차 제곱합 (objective_value): ", result_curvefit.objective_value)
# 최종 잔차 벡터 전체 출력 대신 처음 몇 개만 출력하여 간결하게
println("  최종 잔차 벡터 (residuals, 처음 5개): ", result_curvefit.residuals[1:min(5, end)])


# --- julia_lsqcurvefit 함수 호출 및 결과 출력 (경계 조건 포함) ---
println("\n--- julia_lsqcurvefit 실행 결과 (경계 조건 포함) ---")
lb_exp = [0.0, 0.0] # 파라미터 하한
ub_exp = [20.0, 1.0] # 파라미터 상한

result_curvefit_bounded = julia_lsqcurvefit(model_exp, t_data_exp, y_data_exp, p_init_exp, lb=lb_exp, ub=ub_exp)

# 결과 출력 부분을 보기 좋게 수정
# LsqFit.converged(result.lsqfit_result) 대신 result.lsqfit_result.converged 필드 사용
println("  최적화 상태: ", result_curvefit_bounded.lsqfit_result.converged ? "수렴됨" : "수렴되지 않음")
println("  경계 조건 하의 최적화된 파라미터 (solution): ", result_curvefit_bounded.solution)
println("  최소 잔차 제곱합 (objective_value): ", result_curvefit_bounded.objective_value)
# 최종 잔차 벡터 전체 출력 대신 처음 몇 개만 출력하여 간결하게
println("  최종 잔차 벡터 (residuals, 처음 5개): ", result_curvefit_bounded.residuals[1:min(5, end)])


# (선택 사항) 결과 시각화
# println("\nStarting Plotting...")
# p = scatter(t_data_exp, y_data_exp, label="Data", legend=:topright)
# plot!(p, t_data_exp, model_exp(t_data_exp, result_curvefit.solution), label="Fitted Model (unconstrained)", linewidth=2)
# plot!(p, t_data_exp, model_exp(t_data_exp, result_curvefit_bounded.solution), label="Fitted Model (constrained)", linewidth=2)
# xlabel!("t")
# ylabel!("y")
# title!("Julia lsqcurvefit result")
# display(p) # 플롯 표시
# println("Visualization Completed.")