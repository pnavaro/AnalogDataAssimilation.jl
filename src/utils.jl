using Statistics, LinearAlgebra

"""
    sqrt_svd(A)  

Returns the square root matrix by SVD
"""
function sqrt_svd(A::AbstractMatrix)
    F = svd(A)
    F.U * diagm(sqrt.(F.S)) * F.Vt
end

export RMSE


"""
    RMSE(e)

Returns the Root Mean Squared Error
"""
RMSE(e) = sqrt.(mean(e .^ 2))

"""
    normalise!( w )

Normalize the entries of a multidimensional array sum to 1.
"""
function normalise!(w)

    c = sum(w)
    # Set any zeros to one before dividing
    c += c == 0
    w ./= c

end

""" 
    mk_stochastic!(w)

Ensure the matrix is stochastic, i.e., 
the sum over the last dimension is 1.
"""
function mk_stochastic!(w::Array{Float64,2})

    if first(size(w)) == 1
        normalise!(w)
    else
        # Copy the normaliser plane for each i.
        normaliser = sum(w, dims = 2)
        # Set zeros to 1 before dividing
        # This is valid since normaliser(i) = 0 iff T(i) = 0
        normaliser .+= normaliser .== 0
        w ./= normaliser
    end

end

""" 
    sample_discrete(prob, r, c)

Sampling from a non-uniform distribution. 
"""
function sample_discrete(prob, r, c)

    # this speedup is due to Peter Acklam
    cumprob = cumsum(prob)
    R = rand(r, c)
    M = zeros(Int64, (r, c))
    N = length(cumprob)
    for i = 1:N-1
        M .+= R .> cumprob[i]
    end
    M

end


""" 
    inv_using_SVD(Mat, eigvalMax)

SVD decomposition of Matrix. 
"""
function inv_using_SVD(Mat, eigvalMax)

    F = svd(Mat; full = true)
    eigval = cumsum(F.S) ./ sum(F.S)
    # search the optimal number of eigen values
    icut = findfirst(eigval .>= eigvalMax)

    U_1 = @view F.U[1:icut, 1:icut]
    V_1 = @view F.Vt'[1:icut, 1:icut]
    tmp1 = (V_1 ./ F.S[1:icut]') * U_1'

    if icut + 1 > length(eigval)
        tmp1
    else
        U_3 = @view F.U[icut+1:end, 1:icut]
        V_3 = @view F.Vt'[icut+1:end, 1:icut]
        tmp2 = (V_1 ./ F.S[1:icut]') * U_3'
        tmp3 = (V_3 ./ F.S[1:icut]') * U_1'
        tmp4 = (V_3 ./ F.S[1:icut]') * U_3'
        vcat(hcat(tmp1, tmp2), hcat(tmp3, tmp4))
    end

end

"""
    resample_multinomial( w )

Multinomial resampler. 
"""
function resample_multinomial(w::Vector{Float64})

    m = length(w)
    q = cumsum(w)
    q[end] = 1.0 # Just in case...
    i = 1
    indx = Int64[]
    while i <= m
        sampl = rand()
        j = 1
        while q[j] < sampl
            j = j + 1
        end
        push!(indx, j)
        i = i + 1
    end
    indx
end

""" 
    resample!( indx, w )

Multinomial resampler.
"""
function resample!(indx::Vector{Int64}, w::Vector{Float64})

    m = length(w)
    q = cumsum(w)
    i = 1
    while i <= m
        sampl = rand()
        j = 1
        while q[j] < sampl
            j = j + 1
        end
        indx[i] = j
        i = i + 1
    end
end

"""
    ensure_pos_sym(M; ϵ= 1e-8)

Ensure that matrix `M` is positive and symmetric to avoid numerical errors when numbers are small by doing `(M + M')/2 + ϵ*I`

reference : [StateSpaceModels.jl](https://github.com/LAMPSPUC/StateSpaceModels.jl)
"""
function ensure_pos_sym(M::Matrix{T}; ϵ::T = 1e-8) where {T<:AbstractFloat}
    return (M + M') ./ 2 + ϵ * I
end

function ensure_pos_sym!(M::Matrix{T}; ϵ::T = 1e-8) where {T<:AbstractFloat}
    M .= (M + M') ./ 2 + ϵ * I
end
