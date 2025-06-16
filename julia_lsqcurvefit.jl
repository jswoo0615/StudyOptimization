using LsqFit
using LinearAlgebra # sum(abs2, ...) 사용 가능
# using Plots # 예제에서 사용, Wrapper 함수 자체에는 불필요

"""
julia_lsqcurvefit (LsqFit.jl을 사용한 비선형 모델 피팅)

MATLAB의 lsqcurvefit과 동일한 기능을 제공합니다.
ydata - model_func(xdata, p) 의 잔차 벡터 제곱합을 최소화하는 파라미터 p를 찾습니다.

LsqFit.jl 패키지의 curve_fit 함수를 기반으로 합니다.
 resid 함수를 사용하지 않고 수동으로 잔차를 계산합니다.

인자:
- model_func: 모델 출력을 계산하는 함수. 시그니처는 model_func(xdata, p) 형태여야 합니다.
              xdata는 독립 변수 데이터, p는 최적화할 모델 파라미터 벡터입니다.
- xdata: 독립 변수 데이터 (AbstractArray). MATLAB의 xdata에 해당.
- ydata: 종속 변수 데이터 (AbstractArray). MATLAB의 ydata에 해당.
- p0: 모델 파라미터 벡터의 초기 추정값 (AbstractVector{Float64}). MATLAB의 x0에 해당.
- lb: 파라미터의 하한 벡터 (AbstractVector{Float64} 또는 nothing). 기본값은 nothing (제약 없음). MATLAB의 lb에 해당.
- ub: 파라미터의 상한 벡터 (AbstractVector{Float64} 또는 nothing). 기본값은 nothing (제약 없음). MATLAB의 ub에 해당.
- kwargs: LsqFit.curve_fit 함수에 직접 전달될 추가적인 키워드 인자 딕셔너리.
          예: Dict(:maxIter => 1000, :xtol => 1e-8)

반환 값:
Named Tuple 형태의 결과.
(
    solution: 최적화된 모델 파라미터 벡터 (MATLAB의 x에 해당). coef(result)로 얻음.
    objective_value: 최적화된 파라미터에서의 잔차 제곱합 (MATLAB의 resnorm 또는 fval에 해당). 수동 계산된 잔차 벡터 제곱합.
    residuals: 최적화된 파라미터에서의 잔차 벡터. 수동 계산된 잔차 벡터.
    lsqfit_result: LsqFit.jl의 전체 결과 객체. (수렴 정보 등은 여기서 확인)
)
"""
function julia_lsqcurvefit(model_func, xdata, ydata, p0::AbstractVector{Float64};
                         lb::Union{AbstractVector{Float64}, Nothing}=nothing,
                         ub::Union{AbstractVector{Float64}, Nothing}=nothing,
                         kwargs::Dict{Symbol, Any}=Dict{Symbol, Any}())

    n = length(p0)

    # 입력 유효성 검사
    if length(xdata) != length(ydata)
        error("독립 변수 데이터(xdata)와 종속 변수 데이터(ydata)의 길이가 같아야 합니다.")
    end
    if lb !== nothing && length(lb) != n
        error("하한 벡터 (lb)의 길이가 초기 추정값 (p0)의 길이와 일치해야 합니다.")
    end
    if ub !== nothing && length(ub) != n
        error("상한 벡터 (ub)의 길이가 초기 추정값 (p0)의 길이와 일치해야 합니다.")
    end
     if lb !== nothing && ub !== nothing
         if any(lb .> ub)
             error("하한 (lb) 값은 상한 (ub) 값보다 작거나 같아야 합니다.")
         end
     end


    # LsqFit.curve_fit에 전달할 키워드 인자 준비
    fit_kwargs = Dict{Symbol, Any}()
    if lb !== nothing
        fit_kwargs[:lower] = lb
    end
    if ub !== nothing
        fit_kwargs[:upper] = ub
    end
    merge!(fit_kwargs, kwargs)

    # LsqFit.curve_fit 함수 호출
    fit_result = curve_fit(model_func, xdata, ydata, p0; fit_kwargs...)

    # 결과 추출
    p_fitted = coef(fit_result)                 # 최적화된 파라미터

    # --- 수동 잔차 계산 ---
    # LsqFit.resid 함수 대신, fitted 파라미터와 원본 데이터를 사용하여 직접 잔차 벡터를 계산합니다.
    residual_vec = ydata .- model_func(xdata, p_fitted) # <-- 수동 계산

    fval = sum(abs2, residual_vec)              # 잔차 벡터의 제곱합 계산 (MATLAB의 resnorm 또는 fval)


    # 결과를 Named Tuple 형태로 반환
    return (
        solution=p_fitted,
        objective_value=fval,
        residuals=residual_vec,                 # 수동 계산된 잔차 벡터 반환
        lsqfit_result=fit_result                # LsqFit.jl 결과 객체 전체 (수렴 정보 등은 여기서 확인)
    )
end