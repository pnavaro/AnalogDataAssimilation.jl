using Statistics, LinearAlgebra
using Random

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
function resample_multinomial(w::Vector{Float64}; rng = Random.default_rng())

    m = length(w)
    indx = Vector{Int64}(undef, m)
    resample!(indx, w; rng = rng)
    indx
end

""" 
    resample!( indx, w )

Multinomial resampler.
"""
function resample!(indx::Vector{Int64}, w::Vector{Float64}; rng = Random.default_rng())
    !isempty(w) || throw(ArgumentError("resampling requires at least one weight"))
    all(isfinite, w) || throw(ArgumentError("resampling weights must be finite"))
    all(weight -> weight >= 0.0, w) ||
        throw(ArgumentError("resampling weights must be nonnegative"))
    total = sum(w)
    isfinite(total) && total > 0 ||
        throw(ArgumentError("resampling weights must have a finite positive sum"))

    q = cumsum(w ./ total)
    q[end] = 1.0
    for i in eachindex(indx)
        sample = rand(rng)
        indx[i] = min(searchsortedlast(q, sample) + 1, length(q))
    end
    return indx
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
