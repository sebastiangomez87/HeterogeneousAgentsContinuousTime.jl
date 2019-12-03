using HeterogeneousAgentsContinuousTime
using Test

using Heterogeneous-Agents-Continuous-Time
using Test

results=HJBequation(0.03,0.5)

@test size(results.A)[1]==length(results.a)*length(results.z)
@test results.c[1]≈0.0955
@test results.c[500]≈0.3575
@test results.c[1000]≈0.3612
@test results.c[200]≈0.2478
@test results.v[1]≈-143.49
@test results.v[500]≈-69.23
@test results.v[1000]≈-68.54
@test results.v[200]≈-97.1226

resultsKF=KF(results.a, results.z, results.A)

@test resultsKF[1]≈0
@test resultsKF[10]≈1.44
@test resultsKF[200]≈0
@test resultsKF[500]≈0
@test resultsKF[510]≈1.03
@test resultsKF[550]≈0.68
@test resultsKF[700]≈0

@test equilibriumR(0.5)≈0.02109
@test equilibriumR(0.05)≈0.03619
@test equilibriumR(0.8)≈0.02609
@test equilibriumR(1.0)≈0.0312
@test equilibriumR(1.2)≈0.03419
@test equilibriumR(1.5)≈0.03762

results3=TimeHJBequation(0.03*ones(100),results,100,0.5)

@test results3.v[1,1]≈-143.49
@test results3.v[200,10]≈-97.1227
@test results3.s[300,50]≈-0.11604

results4=TimeKF(results3.a,results3.z,results3.A,resultsKF,0.03*ones(100),20)

@test results4[20,20]≈1.3374
@test results4[20,50]≈1.3374
@test results4[10,20]≈1.43665
@test results4[40,30]≈0.58458

resultsh=HJBequationHousing(0.03,0.5)

@test resultsh.h[200]≈2.3927
@test resultsh.h[1]≈0.0
@test resultsh.h[500]≈5.16269
@test resultsh.v[250]≈-76.4219


@testset "HeterogeneousAgentsContinuousTime.jl" begin
    # Write your own tests here.
end
