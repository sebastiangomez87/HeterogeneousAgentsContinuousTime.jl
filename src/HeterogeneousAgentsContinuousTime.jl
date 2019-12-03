module HeterogeneousAgentsContinuousTime

greet() = print("Hello World!")

using SparseArrays, LinearAlgebra, Plots

function HJBequation(r0, lambda2)
    #This function solves the HJB equation using the implicit method in Achdou et al (2017)
    #r0 is the interest rate and lambda2 the probability to pass to state 2

    #Maximum iterations
    maxiter=100
    #Tolerance
    tol=1e-6
    #Delta to calculate vprime
    delta=100

    #Price
    #r=0.03
    r=r0

    #Discount
    ρ=0.05

    #Parameter for the CRRA
    σ=2.0
    #Inverse of the marginal utility
    uprimeinv(x)=x^(-1/σ)
    #utility
    u(x)=(x^(1-σ))/(1-σ)
    #marginal utility
    uprime(x)=x^(-σ)

    z=[0.1,0.2]#States of the stochastic process
    Nz=length(z)

    #λ=[1.2,1.2]#Transition probabilities
    λ=[0.6,lambda2]#Transition probabilities

    amin=-0.15
    amax=4.0
    Na=500 #number of points in a
    a=[range(amin,stop=amax,length=Na);]
    deltaa=(amax-amin)/(Na-1) #Assumes that a is equally spaced

    #Create big vector with z
    zVec=[]
    for j=1:Nz
        zVec=[zVec;z[j]*ones(Na)]
    end

    v0=zeros(eltype(a),Na*Nz)
    for j=1:Nz
        v0[(j-1)*Na+1:j*Na]=((z[j].+max(r,0.01).*a).^(1-σ))/(1-σ)/ρ
    end

    #definition of the positions that are different from zero in the matrix A
    rows=[1:Na;Na+1:2*Na;1:2*Na;2:Na;Na+2:2*Na;1:Na-1;Na+1:2*Na-1]#Positions of \lambda1 and \lambda 2. Positions of y. Positions for x1. Positions for x2. Positions for z1. Positions for z2
    cols=[Na+1:2*Na;1:Na;1:2*Na;1:Na-1;Na+1:2*Na-1;2:Na;Na+2:2*Na]#Positions of \lambda1 and \lambda 2. Positions of y. Positions for x1. Positions for x2. Positions for z1. Positions for z2
    #Initialization of A
    valsA=ones(eltype(v0),length(rows))
    A=sparse(rows,cols,valsA)

    v1=zeros(eltype(a),Na*Nz)
    c=zeros(eltype(a),Na*Nz)
    s=zeros(eltype(a),Na*Nz)

    for i=1:maxiter
        vprimeF=ones(eltype(a),Na*Nz)
        vprimeB=ones(eltype(a),Na*Nz)
        for j=1:Nz
            vprimeF[(j-1)*Na+1:j*Na-1]=(v0[(j-1)*Na+2:j*Na].-v0[(j-1)*Na+1:j*Na-1])./deltaa #In the last position the forward difference is not defined
            vprimeF[j*Na]=(z[j] + r.*amax).^(-σ) #Will never be used, but just in case
            vprimeB[(j-1)*Na+2:j*Na]=(v0[(j-1)*Na+2:j*Na].-v0[(j-1)*Na+1:j*Na-1])./deltaa #In the first position the backward difference is not defined
            vprimeB[(j-1)*Na+1]=(z[j] + r.*amin).^(-σ) #state constraint boundary condition
        end
        sF=zeros(eltype(vprimeF),Na*Nz)
        sB=zeros(eltype(vprimeB),Na*Nz)

        ######CAREFUL. SOMETIMES VPRIME IS NEGATIVE, THIS ENSURES THIS DOESN'T HAPPEN
        vprimeF[vprimeF.<=0].=tol
        vprimeB[vprimeB.<=0].=tol

        for j=1:Nz
            for k=1:Na
                sF[(j-1)*Na+k]=z[j]+r*a[k]-uprimeinv(vprimeF[(j-1)*Na+k])
                sB[(j-1)*Na+k]=z[j]+r*a[k]-uprimeinv(vprimeB[(j-1)*Na+k])
            end
        end
        vprime=similar(vprimeF)
        for j=1:Nz
            for k=1:Na
                if sF[(j-1)*Na+k]>0
                    vprime[(j-1)*Na+k]=vprimeF[(j-1)*Na+k]
                elseif sB[(j-1)*Na+k]<0
                    vprime[(j-1)*Na+k]=vprimeB[(j-1)*Na+k]
                else
                    vprime[(j-1)*Na+k]=uprime(z[j]+r*a[k])
                end
            end
        end
        #Consumption
        c=uprimeinv.(vprime)

        #Calculation of savings
        s=zVec+r*repeat(a,Nz,1)-c
        b=u.(c)+v0/delta


        x=-sB/deltaa
        x[x.<=0].=0
        w=sF/deltaa #This is the z that goes in the matrix A. It has a conflict with the stochastic process
        w[w.<=0].=0
        #Making the first element of x (for each value of z) 0
        for j=1:Nz
            x[(j-1)*Na+1]=0
        end
        for j=1:Nz
            w[j*Na]=0
        end
        #Definition of y without substracting the probability lambda
        y=-x-w
        #Adjusting y by the probability lambda
        for j=1:Nz
            y[(j-1)*Na+1:j*Na]=y[(j-1)*Na+1:j*Na].-λ[j]
        end

        #Creation of matrix a
        #THIS MATRIX ASSUMES LENGTH(Z)=2
        valsA=vec([λ[1]*ones(eltype(λ[1]),Na,1);λ[2]*ones(eltype(λ[1]),Na,1);y;x[2:Na];x[Na+2:2*Na];w[1:Na-1];w[Na+1:2*Na-1]]) #vector with the values of A. \lambda1 and \lambda 2. Positions of y. Positions for x1. Positions for x2. Positions for z1. Positions for z2
        A=sparse(rows,cols,valsA)

        B=(1/delta+ρ)*I-A
        v1=B\b

        if maximum(abs.(v1-v0))<tol
            break
        end
        v0=copy(v1)
    end

    #p1=plot(a,[s[1:Na] s[Na+1:2*Na]],xlabel="Wealth",title="Savings",label=["y1" "y2"])
    #p2=plot(a,[c[1:Na] c[Na+1:2*Na]],xlabel="Wealth",title="Consumption",label=["y1" "y2"],legend=:topleft)
    #display(plot(p1,p2,layout=(1,2)))

    return (a=a,z=z,c=c,s=s,A=A,v=v0)
end

function KF(a,z,A)
    Na=length(a)
    deltaa=(a[Na]-a[1])/(Na-1)
    Nz=length(z)
    g=zeros(eltype(a),Nz*Na)
    zeroVec=zeros(eltype(a),Nz*Na) #Zero vector in equation 20 of the Numerical appendix
    zeroVec[1]=0.1
    (rowsA,colsA,valsA)=findnz(A) #retrieving rows, columns and values
    m,n=size(A)
    rowsAt=[colsA;ones(n)] #the columns of A and a vector of ones to change first row
    colsAt=[rowsA;collect(1:n)]
    valsAt=[valsA;1;zeros(eltype(a),n-1)]
    At=sparse(rowsAt,colsAt,valsAt)
    gtilda=At\zeroVec
    g=gtilda/(deltaa*sum(gtilda))

    #p1=plot(a[1:Na],g[1:Na],xlabel="Wealth",title="Distribution of agents",label=["y1" "y2"])
    #p2=plot!(a,g[Na+1:2*Na])
    #display(plot(p1,p2))

    return g
end

function AssetSupply()

    #Values to iterate over r
    rmin=-0.05
    rmax=0.04
    Nr=20
    vecr=[range(rmin,stop=rmax,length=Nr);]

    #Supply function
    S=zeros(eltype(vecr),Nr)

    lambda2=0.5

    for i in eachindex(vecr)
        results=HJBequation(vecr[i],lambda2)
        g=KF(results.a,results.z,results.A)
        Na=length(results.a)
        deltaa=(results.a[Na]-results.a[1])/(Na-1)
        S[i]=deltaa*sum(g.*repeat(results.a,2,1))
    end
    plot(S,vecr,xlabel="Net demand", ylabel="r", title="Asset supply", legend=false)
    plot!([0], seriestype="vline")
    #@show vecr
end

function equilibriumR(lambda2)
    #Calculates the equilibrium interest rate

    #Maximum r to consider
    rmax=0.04
    #Minimum r to consider
    rmin=0.01
    #weight of previous r
    wr=0.5

    r0=0.03 #Initial guess

    #Maximum iterations
    itermax=100
    tol=1e-6

    #Excess demand
    S=0

    #Transition probability
    #lambda2=1.2

    for i=1:itermax
        results=HJBequation(r0,lambda2)
        g=KF(results.a,results.z,results.A)
        Na=length(results.a)
        deltaa=(results.a[Na]-results.a[1])/(Na-1)
        S=deltaa*sum(g.*repeat(results.a,2,1))#Total supply
        #@show r0 rmin rmax S
        #@show results.c[1:10]
        if S>tol/100 #&& (rmax-rmin)>tol
            rmax=copy(r0)
            r0=wr*r0+(1-wr)*rmin
        elseif S<-tol/100 #&& (rmax-rmin)>tol #Excess demand
            rmin=copy(r0)
            r0=wr*r0+(1-wr)*rmax
        else
            break
        end
    end
    return r0
    #println(r0, S)
end


function TimeHJBequation(rl, resultsT,T,lambda2)
    #Given a time path for r (rl) between t=0 and t=T, it computes the value and policy functions for each t
    #resultsT are a tuple given by HJBequation which contains the results for the steady state (t=T)
    Nt=length(rl)
    deltat=T/Nt

    #Maximum iterations
    maxiter=100
    #Tolerance
    tol=1e-6

    #Discount
    ρ=0.05

    #Parameter for the CRRA
    σ=2.0
    #Inverse of the marginal utility
    uprimeinv(x)=x^(-1/σ)
    #utility
    u(x)=(x^(1-σ))/(1-σ)
    #marginal utility
    uprime(x)=x^(-σ)

    z=[0.1,0.2]#States of the stochastic process
    Nz=length(z)

    λ=[0.6,lambda2]#Transition probabilities

    amin=-0.15
    amax=4.0
    Na=500 #number of points in a
    a=[range(amin,stop=amax,length=Na);]
    deltaa=(amax-amin)/(Na-1) #Assumes that a is equally spaced

    #Create big vector with z
    zVec=[]
    for j=1:Nz
        zVec=[zVec;z[j]*ones(Na)]
    end

    v=zeros(eltype(a),Na*Nz,Nt+1)

    #Last v is the stationary state value function

    v[:,Nt+1]=resultsT.v



    #definition of the positions that are different from zero in the matrix A
    rows=[1:2*Na;1:2*Na;2:Na;Na+2:2*Na;1:Na-1;Na+1:2*Na-1]#Positions of \lambda1 and \lambda 2. Positions of y. Positions for x1. Positions for x2. Positions for z1. Positions for z2
    cols=[Na+1:2*Na;1:Na;1:2*Na;1:Na-1;Na+1:2*Na-1;2:Na;Na+2:2*Na]#Positions of \lambda1 and \lambda 2. Positions of y. Positions for x1. Positions for x2. Positions for z1. Positions for z2
    #Initialization of A
    valsA=ones(eltype(v),length(rows))
    #In this version A is a vector of sparse matrices
    #A=zeros(2*Na,2*Na,Nt+1)

    A=[]
    push!(A,resultsT.A)
    #A[:,:,Nt+1]=resultsT.A

    #Initialization of Atemp
    #Atemp=sparse(rows,cols,valsA)

    c=zeros(eltype(a),Na*Nz,Nt)
    s=zeros(eltype(a),Na*Nz,Nt)

    for i=Nt:-1:1
        vprimeF=ones(eltype(a),Na*Nz)
        vprimeB=ones(eltype(a),Na*Nz)
        for j=1:Nz
            vprimeF[(j-1)*Na+1:j*Na-1]=(v[(j-1)*Na+2:j*Na,i+1].-v[(j-1)*Na+1:j*Na-1,i+1])./deltaa #In the last position the forward difference is not defined
            vprimeF[j*Na]=(z[j] + rl[i].*amax).^(-σ) #Will never be used, but just in case
            vprimeB[(j-1)*Na+2:j*Na]=(v[(j-1)*Na+2:j*Na,i+1].-v[(j-1)*Na+1:j*Na-1,i+1])./deltaa #In the first position the backward difference is not defined
            vprimeB[(j-1)*Na+1]=(z[j] + rl[i].*amin).^(-σ) #state constraint boundary condition
        end
        sF=zeros(eltype(vprimeF),Na*Nz)
        sB=zeros(eltype(vprimeB),Na*Nz)

        ######CAREFUL. SOMETIMES VPRIME IS NEGATIVE, THIS ENSURES THIS DOESN'T HAPPEN
        vprimeF[vprimeF.<=0].=tol
        vprimeB[vprimeB.<=0].=tol

        for j=1:Nz
            for k=1:Na
                sF[(j-1)*Na+k]=z[j]+rl[i]*a[k]-uprimeinv(vprimeF[(j-1)*Na+k])
                sB[(j-1)*Na+k]=z[j]+rl[i]*a[k]-uprimeinv(vprimeB[(j-1)*Na+k])
            end
        end
        vprime=similar(vprimeF)

        for j=1:Nz
            for k=1:Na
                if sF[(j-1)*Na+k]>0
                    vprime[(j-1)*Na+k]=vprimeF[(j-1)*Na+k]
                elseif sB[(j-1)*Na+k]<0
                    vprime[(j-1)*Na+k]=vprimeB[(j-1)*Na+k]
                else
                    vprime[(j-1)*Na+k]=uprime(z[j]+rl[i]*a[k])
                end
            end
        end
        #Consumption
        c[:,i]=uprimeinv.(vprime)

        #Calculation of savings
        s[:,i]=zVec+rl[i]*repeat(a,Nz,1)-c[:,i]
        b=u.(c[:,i])+v[:,i+1]/deltat


        x=-sB/deltaa
        x[x.<=0].=0
        w=sF/deltaa #This is the z that goes in the matrix A. It has a conflict with the stochastic process
        w[w.<=0].=0
        #Making the first element of x (for each value of z) 0
        for j=1:Nz
            x[(j-1)*Na+1]=0
        end
        for j=1:Nz
            w[j*Na]=0
        end
        #Definition of y without substracting the probability lambda
        y=-x-w
        #Adjusting y by the probability lambda
        for j=1:Nz
            y[(j-1)*Na+1:j*Na]=y[(j-1)*Na+1:j*Na].-λ[j]
        end

        #Creation of matrix a
        #THIS MATRIX ASSUMES LENGTH(Z)=2
        #valsA=vcat(λ[1]*ones(eltype(λ[1]),Na,1),λ[2]*ones(eltype(λ[1]),Na,1),y,x[2:Na],x[Na+2:2*Na],w[1:Na-1],w[Na+1:2*Na-1]) #vector with the values of A. \lambda1 and \lambda 2. Positions of y. Positions for x1. Positions for x2. Positions for z1. Positions for z2
        valsA=vec([λ[1]*ones(eltype(λ[1]),Na,1);λ[2]*ones(eltype(λ[1]),Na,1);y;x[2:Na];x[Na+2:2*Na];w[1:Na-1];w[Na+1:2*Na-1]]) #vector with the values of A. \lambda1 and \lambda 2. Positions of y. Positions for x1. Positions for x2. Positions for z1. Positions for z2
        Atemp=sparse(rows,cols,valsA)

        B=(1/deltat+ρ)*I-Atemp
        v[:,i]=B\b
        pushfirst!(A,Atemp)
    end
    #@show typeof(A[:,:,3])
    return (a=a,z=z,c=c,s=s,A=A[1:end-1],v=v[:,1:Nt])
end


function TimeKF(a,z,A,g0,rl,T)
    #Calculates distribution over time period implied by rl. Taking g0 as initial distribution and A as the matrices at each period in time
    Na=length(a)
    deltaa=(a[Na]-a[1])/(Na-1)
    Nz=length(z)
    Nt=length(rl)
    deltat=T/Nt

    g=zeros(eltype(a),Nz*Na,Nt+1)
    g[:,1]=g0

    for i=2:Nt+1
        Atemp1=A[i-1]
        (rowsA,colsA,valsA)=findnz(Atemp1) #retrieving rows, columns and values
        Atemp=sparse(colsA,rowsA,valsA)#Defining the tranpose
        g[:,i]=(I-deltat*sparse((Atemp)))\g[:,i-1]
    end
    return g
end

function shockSim()
    #Maximum iterations
    maxiter=100
    #Tolerance
    tol=0.001

    #Initial value of lambda2
    lambda2_0=0.5 #1.0
    #Final value of lambda2
    lambda2_T=0.9 #1.5

    r0=equilibriumR(lambda2_0)
    results0=HJBequation(r0, lambda2_0)
    a=results0.a
    Na=length(a)
    deltaa=(a[end]-a[1])/(Na-1)

    rT=equilibriumR(lambda2_T)
    resultsT=HJBequation(rT, lambda2_T)

    g0=KF(results0.a,results0.z,results0.A)
    gT=KF(resultsT.a,resultsT.z,resultsT.A)

    T=20
    #Number of points in the time interval
    N=100

    #Matrix to save g
    g=repeat(g0,1,N+1)

    #speed of updating the interest rate. Taken directly from  Moll's code
    xi = 20*(exp.(-0.05*(1:N)) .- exp.(-0.05*N))

    #First guess on the path of r
    rl0=rT*ones(eltype(rT),N)
    rl1=copy(rl0)
    @show r0 rT

    for i=1:maxiter
        results=TimeHJBequation(rl0, resultsT,T, lambda2_T)
        g=TimeKF(results0.a,results0.z,results.A,g0,rl0,T)
        S=deltaa.*sum(g.*repeat(a,2,N+1),dims=1)
        diffS=S[2:end]-S[1:end-1]
        #@show maximum(abs.(diffS))
        rl1=rl0-xi.*diffS
        #@show maximum(abs.(rl1-rl0))
        if maximum(abs.(diffS))<tol || maximum(abs.(rl1-rl0))<tol
            break
        end
        #println(rl1)
        rl0=copy(rl1)
    end
    #@show rl0
    p1=plot([range(-5*T/N,stop=T,length=N+5);],[r0*ones(5); rl1],xlabel="T",ylabel="r",legend=false)
    p2=plot(a[1:100], [g[1:100,1] g[Na+1:Na+100,1] g[1:100,3] g[Na+1:Na+100,3]],xlabel="Wealth",title="t=0.06",label=["y1 t=0" "y2  t=0" "y1 t=0.06" "y2  t=0.06"],linecolor=["red" "blue" "red" "blue"],linestyle=[:solid :solid :dash :dash])
    p3=plot(a[1:100], [g[1:100,1] g[Na+1:Na+100,1] g[1:100,N+1] g[Na+1:Na+100,N+1]],xlabel="Wealth",title="Long run",label=["y1 t=0" "y2  t=0" "y1 t=T" "y2  t=T"],linecolor=["red" "blue" "red" "blue"],linestyle=[:solid :solid :dash :dash])
    display(plot(p1,p2,p3,layout=(1,3)))
end

function optimalh(a, ϕ, r, p, h,f)
    #Finds the optimal h given a value of wealth a. In this case the input a is a value, while h is a vector of possible values of h.
    #From the paper it should have this shape={0,[hmin,\infty]}
    #f is the function of housing in the utility function

    #Possible values of utility for all h
    temp=f.(h)
    temp[p*h.>ϕ*a].=-Inf
    #Position of the optimal h
    hopt=argmax(temp)
    #Value at optimum
    ftilda=f(h[hopt])

    return (ftilda,hopt)

end


function HJBequationHousing(r0, lambda2)
        #This function solves the HJB equation using the implicit method in Achdou et al (2017)
        #r0 is the interest rate and lambda2 the probability to pass to state 2

        #Maximum iterations
        maxiter=120
        #Tolerance
        tol=1e-6
        #Delta to calculate vprime
        delta=1000

        #Price
        #r=0.03
        r=r0

        #Discount
        ρ=0.05

        #Parameter for the CRRA
        σ=2.0
        #Inverse of the marginal utility
        uprimeinv(x)=x^(-1/σ)
        #utility
        u(x)=(x^(1-σ))/(1-σ)
        #marginal utility
        uprime(x)=x^(-σ)

        z=[0.1,0.135]#States of the stochastic process
        Nz=length(z)

        #λ=[1.2,1.2]#Transition probabilities
        λ=[0.5,lambda2]#Transition probabilities

        amin=0.0
        amax=3.0
        Na=500 #number of points in a
        a=[range(amin,stop=amax,length=Na);]
        deltaa=(amax-amin)/(Na-1) #Assumes that a is equally spaced

        #Create big vector with z
        zVec=[]
        for j=1:Nz
            zVec=[zVec;z[j]*ones(Na)]
        end

        #Housing parameters
        hmin=2.3
        p=1
        ϕ=2
        #Parameters of f
        α=1/3
        η=0.2
        #Utility of housing
        f(x)=η*(max(x-hmin,0))^α - r*p*x

        #Vector with possible values of housing
        Nh=10000
        h=[0;range(hmin,stop=amax*ϕ/p,length=(Nh-1));]

        #Finding optimal values for h given each value of a
        hopt=zeros(Float64,Na)
        fopt=zeros(eltype(h),Na)
        hopt=min.((r*p/(α*η))^(1/(α-1))+hmin,ϕ*a./p)
        hopt[hopt.<hmin].=0
        fopt=f.(hopt)
        #for i=1:Na
        #    (fopt[i],hopt[i])=optimalh(a[i], ϕ, r, p, h,f)
        #end
        #@show hopt
        v0=zeros(eltype(a),Na*Nz)
        for j=1:Nz
            v0[(j-1)*Na+1:j*Na]=((z[j].+max(r,0.01).*a).^(1-σ))/(1-σ)/ρ
        end

        #definition of the positions that are different from zero in the matrix A
        rows=[1:Na;Na+1:2*Na;1:2*Na;2:Na;Na+2:2*Na;1:Na-1;Na+1:2*Na-1]#Positions of \lambda1 and \lambda 2. Positions of y. Positions for x1. Positions for x2. Positions for z1. Positions for z2
        cols=[Na+1:2*Na;1:Na;1:2*Na;1:Na-1;Na+1:2*Na-1;2:Na;Na+2:2*Na]#Positions of \lambda1 and \lambda 2. Positions of y. Positions for x1. Positions for x2. Positions for z1. Positions for z2
        #Initialization of A
        valsA=ones(eltype(v0),length(rows))
        A=sparse(rows,cols,valsA)

        v1=zeros(eltype(a),Na*Nz)
        cx=zeros(eltype(a),Na*Nz)#This is defined as c+f(h)
        c=zeros(eltype(a),Na*Nz)
        s=zeros(eltype(a),Na*Nz)

        for i=1:maxiter
            vprimeF=ones(eltype(a),Na*Nz)
            vprimeB=ones(eltype(a),Na*Nz)
            for j=1:Nz
                vprimeF[(j-1)*Na+1:j*Na-1]=(v0[(j-1)*Na+2:j*Na].-v0[(j-1)*Na+1:j*Na-1])./deltaa #In the last position the forward difference is not defined
                vprimeF[j*Na]=(z[j] + fopt[Na] +r.*amax).^(-σ) #Will never be used, but just in case
                vprimeB[(j-1)*Na+2:j*Na]=(v0[(j-1)*Na+2:j*Na].-v0[(j-1)*Na+1:j*Na-1])./deltaa #In the first position the backward difference is not defined
                vprimeB[(j-1)*Na+1]=(z[j] + fopt[1] +r.*amin).^(-σ) #state constraint boundary condition
            end
            sF=zeros(eltype(vprimeF),Na*Nz)
            sB=zeros(eltype(vprimeB),Na*Nz)

            ######CAREFUL. SOMETIMES VPRIME IS NEGATIVE, THIS ENSURES THIS DOESN'T HAPPEN
            vprimeF[vprimeF.<=0].=10^(-10)
            vprimeB[vprimeB.<=0].=10^(-10)

            for j=1:Nz
                for k=1:Na
                    #@show j k
                    sF[(j-1)*Na+k]=z[j]+fopt[k]+r*a[k]-uprimeinv(vprimeF[(j-1)*Na+k])
                    sB[(j-1)*Na+k]=z[j]+fopt[k]+r*a[k]-uprimeinv(vprimeB[(j-1)*Na+k])
                end
            end
            vprime=similar(vprimeF)
            #vprime for the first value
            #for j=1:Nz
            #    if sF[(j-1)*Na+1]>0
            #        vprime[(j-1)*Na+1]=vprimeF[(j-1)*Na+1]
            #    else
            #        vprime[(j-1)*Na+1]=uprime(z[j]+r*amin)
            #    end
            #end
            #vprime for the last value
            #for j=1:Nz
            #    if sB[j*Na]<0
            #        vprime[j*Na]=vprimeB[j*Na]
            #    else
            #        vprime[j*Na]=uprime(z[j]+r*amax)
            #    end
            #end
            #vprime for intermediate values
            indexB=zeros(Int64,Na*Nz) #1 if using backward
            indexF=zeros(Int64,Na*Nz) #1 if using forward
            for j=1:Nz
                for k=1:Na
                    #utility plus continuation value forward
                    uF=u(uprimeinv(vprimeF[(j-1)*Na+k]))+vprimeF[(j-1)*Na+k]*sF[(j-1)*Na+k]
                    uB=u(uprimeinv(vprimeB[(j-1)*Na+k]))+vprimeB[(j-1)*Na+k]*sB[(j-1)*Na+k]
                    if ((sF[(j-1)*Na+k]>0) & (sB[(j-1)*Na+k]>=0)) | ((sF[(j-1)*Na+k]>0) & (sB[(j-1)*Na+k]<0) & (uF>=uB))
                        vprime[(j-1)*Na+k]=vprimeF[(j-1)*Na+k]
                        indexF[(j-1)*Na+k]=1
                    elseif ((sB[(j-1)*Na+k]<0) & (sF[(j-1)*Na+k]<=0)) | ((sB[(j-1)*Na+k]<0) & (sF[(j-1)*Na+k]>0) & (uB>=uF))
                        vprime[(j-1)*Na+k]=vprimeB[(j-1)*Na+k]
                        indexB[(j-1)*Na+k]=1
                    else
                        vprime[(j-1)*Na+k]=uprime(z[j]+fopt[k]+r*a[k])
                    end
                end
            end
            #Consumption
            cx=uprimeinv.(vprime)

            #Calculation of savings
            s=zVec+repeat(fopt,Nz,1)+r*repeat(a,Nz,1)-cx
            b=u.(cx)+v0/delta


            x=-sB/deltaa
            x[indexB.==0].=0
            #@show x
            w=sF/deltaa #This is the z that goes in the matrix A. It has a conflict with the stochastic process
            w[indexF.==0].=0
            #Making the first element of x (for each value of z) 0
            for j=1:Nz
                x[(j-1)*Na+1]=0
            end
            for j=1:Nz
                w[j*Na]=0
            end
            #Definition of y without substracting the probability lambda
            y=-x-w
            #Adjusting y by the probability lambda
            for j=1:Nz
                y[(j-1)*Na+1:j*Na]=y[(j-1)*Na+1:j*Na].-λ[j]
            end

            #Creation of matrix a
            #THIS MATRIX ASSUMES LENGTH(Z)=2
            #valsA=vcat(λ[1]*ones(eltype(λ[1]),Na,1),λ[2]*ones(eltype(λ[1]),Na,1),y,x[2:Na],x[Na+2:2*Na],w[1:Na-1],w[Na+1:2*Na-1]) #vector with the values of A. \lambda1 and \lambda 2. Positions of y. Positions for x1. Positions for x2. Positions for z1. Positions for z2
            valsA=vec([λ[1]*ones(eltype(λ[1]),Na,1);λ[2]*ones(eltype(λ[1]),Na,1);y;x[2:Na];x[Na+2:2*Na];w[1:Na-1];w[Na+1:2*Na-1]]) #vector with the values of A. \lambda1 and \lambda 2. Positions of y. Positions for x1. Positions for x2. Positions for z1. Positions for z2
            A=sparse(rows,cols,valsA)

            B=(1/delta+ρ)*I-A
            v1=B\b

            if maximum(abs.(v1-v0))<tol
                break
            end
            v0=copy(v1)
        end

        c=cx.-repeat(η*(max.(hopt.-hmin,0)).^α,Nz)

        #p1=plot(a,[s[1:Na] s[Na+1:2*Na]],xlabel="Wealth",title="Savings",label=["y1" "y2"],legend=:topright)
        #p2=plot(a,[v1[1:Na] v1[Na+1:2*Na]],legend=false)
        #p3=plot(a, hopt,xlabel="Wealth",title="Housing",legend=false)
        #p4=plot(a,[c[1:Na] c[Na+1:2*Na]],xlabel="Wealth",title="Consumption",label=["y1" "y2"],legend=:bottomright)
        #p5=plot(a,fopt,legend=false)
        #display(plot(p1,p3,p4,layout=(1,3)))
        #@show p*hmin/ϕ
        #@show h[hopt]
        return (a=a,z=z,c=c,s=s,h=hopt,A=A,v=v0)
end

function KFHousing(a,z,A)
    Na=length(a)
    deltaa=(a[Na]-a[1])/(Na-1)
    Nz=length(z)
    g=ones(eltype(a),Nz*Na)
    g=g/(deltaa*sum(g))
    gtilda=copy(g)
    N=1000#Number of times until convergence to distribution
    dt=10
    At=A'
    for i=1:N
        gtilda=(I - At*dt)\g
        g=copy(gtilda)
    end
    #g=gtilda/(deltaa*sum(gtilda))
    p1=plot(a[1:Na],g[1:Na],xlabel="Wealth",title="Distribution of agents",label=["y1" "y2"])
    p2=plot!(a,g[Na+1:2*Na])
    display(plot(p1,p2))

    return g
end

end # module
