export AnEnKS

struct AnEnKS

    np::Int64

end

""" 
    data_assimilation( yo, da)

Apply stochastic and sequential data assimilation technics using 
model forecasting or analog forecasting. 
"""
function forecast(da::DataAssimilation, yo::TimeSeries, mc::AnEnKS; progress = true)

    # dimensions
    nt = yo.nt        # number of observations
    np = mc.np        # number of particles
    nv = yo.nv        # number of variables (dimensions of problem)

    # initialization
    x̂ = TimeSeries(nt, nv)

    m_xa_part = [zeros(Float64, (nv, np)) for i = 1:nt]
    xf_part = [zeros(Float64, (nv, np)) for i = 1:nt]
    part = [zeros(Float64, (nv, np)) for i = 1:nt]
    pf = [zeros(Float64, (nv, nv)) for i = 1:nt]
    xf = zeros(Float64, (nv, np))
    m_xa_part_tmp = similar(xf)
    xf_mean = similar(xf)
    ef = similar(xf)
    Ks = zeros(Float64, (nv, nv))

	rng = MersenneTwister(1234)

    if progress
        p = Progress(nt)
    end

    sampler1 = MvNormal(da.xb, da.B)

    for k = 1:nt
        if progress
            next!(p)
        end

        # update step (compute forecasts)            
        if k == 1
            rand!(rng, sampler1, xf)
        else
            xf, xf_mean = da.m(part[k-1])
            m_xa_part_tmp .= xf_mean
            m_xa_part[k] .= xf_mean
        end

		xf_part[k] .= xf

        ef .= xf * (Matrix(I, np, np) .- 1 / np)
		pf[k] .= (ef * ef') ./ (np - 1)

        # analysis step (correct forecasts with observations)          
        ivar_obs = findall(.!isnan.(yo.u[k]))
        n = length(ivar_obs)

        if n > 0

            μ = zeros(Float64, n)
            σ = da.R[ivar_obs, ivar_obs]
            sampler2 = MvNormal(μ, σ)
            eps = rand(rng, sampler2, np)
            yf = da.H[ivar_obs, :] * xf

            Σ = (da.H[ivar_obs, :] * pf[k]) * da.H[ivar_obs, :]'
            Σ .+= da.R[ivar_obs, ivar_obs]

			invΣ = inv(Σ)
			K = (pf[k] * da.H[ivar_obs, :]') * invΣ
            d = yo.u[k][ivar_obs] .- yf .+ eps
            part[k] .= xf .+ K * d
			# compute likelihood
			innov_ll = vec(mean(yo.u[k][ivar_obs] .- yf, dims=2))
            loglik = -0.5 * innov_ll'invΣ * innov_ll .- 0.5*(n*log(2π)+log(det(Σ)))

        else

			part[k] .= xf

        end

        x̂.u[k] .= vec(sum(part[k] ./ np, dims = 2))

    end

    if progress
        p = Progress(nt)
    end

    for k = nt:-1:1
        if progress
            next!(p)
        end

        if k == nt
            part[k] .= part[nt]
        else
			m_xa_part_tmp = m_xa_part[k+1]
			tej, m_xa_tmp = da.m(mean(part[k], dims = 2))
            tmp1 = part[k] .- mean(part[k], dims = 2)
            tmp2 = m_xa_part_tmp .- m_xa_tmp
            Ks .= ((tmp1 * tmp2') * pinv(pf[k+1], rtol = 1e-6)) ./ (np - 1)
            part[k] .+= Ks * (part[k+1] .- xf_part[k+1])
        end
        x̂.u[k] .= vec(sum(part[k] ./ np, dims = 2))

    end

    x̂

end
