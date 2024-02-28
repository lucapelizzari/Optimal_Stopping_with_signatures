import numpy as np
import scipy.interpolate
import scipy.special
import ComputationalFinance as cf
import scipy.stats
from scipy.special import ndtri
from scipy.stats.qmc import Sobol
from numpy.random import default_rng
import psutil


def get_necessary_memory(N, return_times, m):
    """
    Estimates the (square root of the) number of bytes needed to simulate the required number of paths.
    :param N: Dimension of the Markovian approximation. 0 if only the stock is saved
    :param return_times: Number of time steps that should be returned
    :param m: Number of sample paths
    :return: Square root of the number of bytes that are required for storing m paths with return_times time steps
        where the volatility has N dimensions
    """
    return 2.5 * np.sqrt(N + 2) * np.sqrt(return_times + 1) * np.sqrt(m) * np.sqrt(np.array([0.]).nbytes)


def get_n_batches(N, return_times, m):
    """
    Returns the number of batches that need to be used to simulate m samples.
    :param N: Dimension of the Markovian approximation. 0 if only the stock is saved
    :param return_times: Number of time steps that should be returned
    :param m: Number of sample paths
    :return: Number of batches and number of samples per batch
    """
    available_memory = np.sqrt(psutil.virtual_memory().available) / 2
    necessary_memory = get_necessary_memory(N=N, return_times=return_times, m=m)
    n_batches = int(np.ceil((necessary_memory / available_memory) ** 2))
    m_batch = int(np.ceil(m / n_batches))
    return n_batches, m_batch


def samples(lambda_, nu, theta, V_0, T, nodes, weights, rho, S_0, r, m, N_time, sample_paths=False, return_times=None,
            vol_only=False, stock_only=False, euler=False, qmc=False, rng=None, rv_shift=False, verbose=0):
    """
    Simulates sample paths under the Markovian approximation of the rough Heston model.
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_time: Number of time steps for the simulation
    :param S_0: Initial stock price
    :param r: Interest rate
    :param m: Number of samples
    :param nodes: Nodes of the Markovian approximation
    :param weights: Weights of the Markovian approximation
    :param sample_paths: If True, returns the entire sample paths, not just the final values. Also returns the sample
        paths of the square root of the volatility and the components of the volatility
    :param return_times: Integer that specifies how many time steps are returned. Only relevant if sample_paths is True.
        E.g., N_time is 100 and return_times is 25, then the paths are simulated using 100 equispaced time steps, but
        only the 26 = 25 + 1 values at the times np.linspace(0, T, 26) are returned. May be used especially for storage
        saving reasons, as only these (in this case 26) values are ever stored. The number N_time must be divisible by
        return_times. If return_times is None, it is set to N_time, i.e. we return every time step that was simulated.
    :param vol_only: If True, simulates only the volatility process, not the stock price process
    :param stock_only: If True, returns only the stock price process. This saves memory
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param rng: Can specify a sampler to use for sampling the underlying random variables. If qmc is true, expects
        an instance of scipy.stats.qmc.Sobol() with the correctly specified dimension of the simulated random variables.
        If qmc is False, expects an instance of np.random.default_rng()
    :param rv_shift: Only relevant when using QMC. Can specify a shift by which the uniform random variables in [0,1)^d
        are drawn. When the random variables X are drawn from Sobol, instead uses (X + rv_shift) mod 1. If True,
        randomly generates such a random shift
    :param verbose: Determines the number of intermediary results printed to the console
    :return: Numpy array of the simulations, and the rng that was used for generating the underlying random variables
    """
    if sample_paths is False:
        return_times = 1
    if return_times is None:
        return_times = N_time
    if N_time % return_times != 0:
        raise ValueError(f'The number of time steps for the simulation N_time={N_time} is not divisible by the number'
                         f'of time steps that should be returned return_times={return_times}.')
    saving_steps = N_time // return_times
    dt = T / N_time
    N = len(nodes)
    if N == 1:
        nodes = np.array([nodes[0], 2 * nodes[0] + 1])
        weights = np.array([weights[0], 0])
        N = 2
        one_node = True
    else:
        one_node = False

    if rng is None:
        if qmc:
            if vol_only:
                rng = Sobol(d=N_time, scramble=False)
            else:
                if euler:
                    rng = Sobol(d=2 * N_time, scramble=False)
                else:
                    rng = Sobol(d=3 * N_time, scramble=False)
        else:
            rng = np.random.default_rng()
    if isinstance(rv_shift, bool) and rv_shift:
        dim = N_time
        if not vol_only and euler:
            dim = 2 * N_time
        elif not vol_only and not euler:
            dim = 3 * N_time
        rv_shift = np.random.uniform(0, 1, dim)

    m_input = m  # the original input of how many samples should be simulated
    # m itself is the number of samples that we simulate
    # We always have m >= m_input. At the end, we discard the additionally simulated paths.

    """
    if qmc:
        if int(2 ** np.ceil(np.log2(m)) + 0.001) != m_input:
            print(f'Using QMC requires simulating a number m of samples that is a power of 2. The input m={m_input} '
                  f'is not a power of 2.')
    """

    available_memory = np.sqrt(psutil.virtual_memory().available)
    necessary_memory = get_necessary_memory(N=0 if stock_only else N, return_times=return_times, m=m)
    if necessary_memory > available_memory:
        raise MemoryError(f'Not enough memory to store the sample paths of the rough Heston model with '
                          f'{N} Markovian dimensions, {return_times} time points where the sample paths should be '
                          f'returned and {m} sample paths. Roughly {necessary_memory}**2 bytes needed, '
                          f'while only {available_memory}**2 bytes are available.')

    available_memory_for_random_variables = available_memory / 3
    if qmc:
        necessary_memory_for_random_variables = np.sqrt(3 * N_time) * np.sqrt(m) * np.sqrt(np.array([0.]).nbytes)
    else:
        necessary_memory_for_random_variables = np.sqrt(3 * m) * np.sqrt(np.array([0.]).nbytes)
    n_batches = int(np.ceil((necessary_memory_for_random_variables / available_memory_for_random_variables) ** 2))
    m_batch = int(np.ceil(m / n_batches))
    m = m_batch * n_batches

    V_init = V_0 / nodes / (np.sum(weights / nodes))

    if euler:
        A = np.eye(N) + np.diag(nodes) * dt + lambda_ * weights[None, :] * dt
        A_inv = np.linalg.inv(A)
        b = theta * dt + (nodes * V_init)[:, None] * dt

        if vol_only:
            def step_SV(V_comp_, dW_):
                sq_V = np.sqrt(np.fmax(weights @ V_comp_, 0))
                # dW_samples = cf.rand_normal(loc=0, scale=np.sqrt(dt), size=m, antithetic=antithetic)
                return A_inv @ (V_comp_ + nu * (sq_V * dW_)[None, :] + b)
        else:
            def step_SV(log_S_, V_comp_, dBW_):
                sq_V = np.sqrt(np.fmax(weights @ V_comp_, 0))
                # dW = cf.rand_normal(loc=0, scale=np.sqrt(dt), size=m, antithetic=antithetic)
                # dB = cf.rand_normal(loc=0, scale=np.sqrt(dt), size=m, antithetic=antithetic)
                log_S_ = log_S_ + r * dt + sq_V * (rho * dBW_[:, 1] + np.sqrt(1 - rho ** 2) * dBW_[:, 0]) \
                    - 0.5 * sq_V ** 2 * dt
                V_comp_ = A_inv @ (V_comp_ + nu * (sq_V * dBW_[:, 1])[None, :] + b)
                return log_S_, V_comp_

    else:
        weight_sum = np.sum(weights)
        A = -(np.diag(nodes) + lambda_ * weights[None, :]) * dt / 2
        exp_A = scipy.linalg.expm(A)
        b = (nodes * V_init + theta) * dt / 2
        ODE_b = np.linalg.solve(A, (exp_A - np.eye(N)) @ b)[:, None]
        z = weight_sum ** 2 * nu ** 2 * dt
        rho_bar_sq = 1 - rho ** 2
        rho_bar = np.sqrt(rho_bar_sq)

        def ODE_step_V(V_):
            return exp_A @ V_ + ODE_b

        B = (6 + np.sqrt(3)) / 4
        A = B - 0.75

        def SDE_step_V(V_, dW_):
            x = weights @ V_
            # dW_ = cf.rand_uniform(size=m, antithetic=antithetic)
            temp = np.sqrt((3 * z) * x + (B * z) ** 2)
            p_1 = (z / 2) * x * ((A * B - A - B + 1.5) * z + (np.sqrt(3) - 1) / 4 * temp + x) / (
                    (x + B * z - temp) * temp * (temp - (B - A) * z))
            p_2 = x / (1.5 * x + A * (B - A / 2) * z)
            test_1 = dW_ < p_1
            test_2 = p_1 + p_2 <= dW_
            x_step = A * z * np.ones(len(temp))
            x_step[test_1] = B * z - temp[test_1]
            x_step[test_2] = B * z + temp[test_2]
            return V_ + (x_step / weight_sum)[None, :]

        def step_V(V_, dW_):
            return ODE_step_V(SDE_step_V(ODE_step_V(V_), dW_))

        def SDE_step_B(log_S_, V_, dB_):
            # dB_ = cf.rand_normal(loc=0, scale=np.sqrt(dt / 2), size=m, antithetic=antithetic)
            x = weights @ V_
            return log_S_ + np.sqrt(x) * rho_bar * dB_ - (0.5 * rho_bar_sq * dt / 2) * x, V_

        drift_SDE_step_W = - (nodes[0] * V_init[0] + theta) * dt
        fact_1 = dt / 2 * (lambda_ - 0.5 * rho * nu)

        def SDE_step_W(log_S_, V_, dW_):
            V_new = step_V(V_, dW_)
            dY = V_ + V_new
            log_S_new = log_S_ + r * dt + rho / nu * (drift_SDE_step_W + (dt / 2 * nodes[0]) * dY[0, :]
                                                      + fact_1 * (weights @ dY) + (V_new[0, :] - V_[0, :]))
            return log_S_new, V_new

        if vol_only:
            def step_SV(V_, dW_):
                return step_V(V_, dW_)
        else:
            def step_SV(S_, V_, dBW_):
                return SDE_step_B(*SDE_step_W(*SDE_step_B(S_, V_, dBW_[:, 0]), dBW_[:, 2]), dBW_[:, 1])

    def generate_samples():
        if vol_only:
            if qmc:
                rv = rng.random(n=m_batch)
                if isinstance(rv_shift, np.ndarray):
                    rv = (rv + rv_shift) % 1.
                if euler:
                    if rv[0, 0] == 0.:
                        rv[1:, :] = np.sqrt(dt) * ndtri(rv[1:, :])  # first sample is 0 and cannot be inverted
                        rv[0, :] = 0.
                    else:
                        rv = np.sqrt(dt) * ndtri(rv)
                return lambda index: rv[:, index]
            else:
                if euler:
                    return lambda index: np.sqrt(dt) * rng.standard_normal(m_batch)
                else:
                    return lambda index: rng.uniform(0, 1, m_batch)
        else:
            if qmc:
                rv = rng.random(n=m_batch)
                if isinstance(rv_shift, np.ndarray):
                    rv = (rv + rv_shift) % 1.
                if euler:
                    if rv[0, 0] == 0.:
                        rv[1:, :] = np.sqrt(dt) * ndtri(rv[1:, :])  # first sample is 0 and cannot be inverted
                        rv[0, :] = 0.
                    else:
                        rv = np.sqrt(dt) * ndtri(rv)
                else:
                    if rv[0, 0] == 0.:  # first sample is 0 and cannot be inverted
                        rv[1:, :2 * N_time] = np.sqrt(dt / 2) * ndtri(rv[1:, :2 * N_time])
                        rv[0, :2 * N_time] = 0.
                    else:
                        rv[:, :2 * N_time] = np.sqrt(dt / 2) * ndtri(rv[:, :2 * N_time])
                return lambda index: rv[:, index::N_time]
            else:
                if euler:
                    return lambda index: np.sqrt(dt) * rng.standard_normal((m_batch, 2))
                else:
                    def rv_fun(index):
                        rv_vars = np.empty((m_batch, 3))
                        rv_vars[:, :2] = np.sqrt(dt / 2) * rng.standard_normal((m_batch, 2))
                        rv_vars[:, 2] = rng.uniform(0, 1, m_batch)
                        return rv_vars
                    return rv_fun

    if vol_only:
        result = np.empty((N + 1, return_times + 1, m)) if sample_paths else np.empty((N + 1, m))
        for j in range(n_batches):
            dW = generate_samples()
            current_V_comp = np.empty((N, m_batch))
            current_V_comp[:, :] = V_init[:, None]
            for i in range(N_time):
                if verbose >= 1:
                    print(f'Simulation round {j + 1} of {n_batches}, step {i + 1} of {N_time}')
                current_V_comp = step_SV(current_V_comp, dW(i))
                if sample_paths and (i + 1) % saving_steps == 0:
                    result[1:, (i + 1) // saving_steps, j * m_batch:(j + 1) * m_batch] = current_V_comp
            if not sample_paths:
                result[1:, j * m_batch:(j + 1) * m_batch] = current_V_comp

        if sample_paths:
            result[1:, 0, :] = V_init[:, None]
            result[0, :, :] = np.fmax(np.einsum('i,ijk->jk', weights, result[1:, :, :]), 0)
        else:
            result[0, :] = np.fmax(np.einsum('i,ij->j', weights, result[1:, :]), 0)
    elif stock_only:
        result = np.empty((return_times + 1, m)) if sample_paths else np.empty(m)
        for j in range(n_batches):
            dBW = generate_samples()
            current_V_comp = np.empty((N, m_batch))
            current_V_comp[:, :] = V_init[:, None]
            current_log_S = np.full(m_batch, np.log(S_0))
            for i in range(N_time):
                if verbose >= 1:
                    print(f'Simulation round {j + 1} of {n_batches}, step {i + 1} of {N_time}')
                current_log_S, current_V_comp = step_SV(current_log_S, current_V_comp, dBW(i))
                if sample_paths and (i + 1) % saving_steps == 0:
                    result[(i + 1) // saving_steps, j * m_batch:(j + 1) * m_batch] = current_log_S
            if not sample_paths:
                result[j * m_batch:(j + 1) * m_batch] = current_log_S
        if sample_paths:
            result[0, :] = np.log(S_0)
    else:
        result = np.empty((N + 2, return_times + 1, m)) if sample_paths else np.empty((N + 2, m))
        for j in range(n_batches):
            dBW = generate_samples()
            current_V_comp = np.empty((N, m_batch))
            current_V_comp[:, :] = V_init[:, None]
            current_log_S = np.full(m_batch, np.log(S_0))
            for i in range(N_time):
                if verbose >= 1:
                    print(f'Simulation round {j + 1} of {n_batches}, step {i + 1} of {N_time}')
                current_log_S, current_V_comp = step_SV(current_log_S, current_V_comp, dBW(i))
                if sample_paths and (i + 1) % saving_steps == 0:
                    result[0, (i + 1) // saving_steps, j * m_batch:(j + 1) * m_batch] = current_log_S
                    result[2:, (i + 1) // saving_steps, j * m_batch:(j + 1) * m_batch] = current_V_comp
            if not sample_paths:
                result[0, j * m_batch:(j + 1) * m_batch] = current_log_S
                result[2:, j * m_batch:(j + 1) * m_batch] = current_V_comp
        if sample_paths:
            result[0, 0, :] = np.log(S_0)
            result[2:, 0, :] = V_init[:, None]
            result[1, :, :] = np.fmax(np.einsum('i,ijk->jk', weights, result[2:, :, :]), 0)
        else:
            result[1, :] = np.fmax(np.einsum('i,ij->j', weights, result[2:, :]), 0)

    result = result[..., :m_input]
    if stock_only:
        result = np.exp(result)
    else:
        result[0, ...] = np.exp(result[0, ...])
        if one_node:
            result = result[:-1, ...]
    return result, rng


def eur(K, lambda_, rho, nu, theta, V_0, S_0, T, nodes, weights, r, m, N_time, n_maturities=None, euler=False, qmc=True,
        payoff='call', implied_vol=False, qmc_error_estimators=25, verbose=0):
    """
    Gives the price or the implied volatility of a European option in the approximated, Markovian rough Heston model
    using MC or QMC simulation.
    :param K: Strike prices, assumed to be a 1d numpy array
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param T: Maturity. Is an integer. For surfaces/multiple maturities, additionally use the parameter n_maturities
    :param nodes: The nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param r: Interest rate
    :param m: Number of samples
    :param N_time: Number of time steps used in simulation
    :param n_maturities: If None, only uses the maturity T. If an integer, uses the maturity vector
        np.linspace(0, T, n_maturities + 1)[1:]. The strikes K are rescaled accordingly, where it is assumed that the
        given vector K corresponds to the largest maturity T. If n_maturities is an integer, N_time must be divisible by
        n_maturities
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param payoff: The payoff function, or the string 'call' or the string 'put'
    :param implied_vol: If True (only for payoff 'call' or 'put') returns the implied volatility, else returns the price
    :param qmc_error_estimators: Runs the pricing step of the Longstaff-Schwartz algorithm qmc_error_estimators times
        to get an MC estimate for the QMC error
    :param verbose: Determines the number of intermediary results printed to the console
    return: The prices of the call option for the various strike prices in K
    """
    is_smile = n_maturities is None
    if is_smile:
        n_maturities = 1
    T_vec = np.linspace(0, T, n_maturities + 1)[1:]
    K_mat = S_0 * np.exp(np.sqrt(T_vec[:, None] / T) * np.log(K / S_0)[None, :])

    def get_samples(rv_shift=False):
        samples_, rng_ = samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, nodes=nodes, weights=weights,
                                 rho=rho, S_0=S_0, r=r, m=m, N_time=N_time, sample_paths=True,
                                 return_times=n_maturities, vol_only=False, stock_only=True, euler=euler, qmc=qmc,
                                 rv_shift=rv_shift, verbose=verbose - 1)
        return samples_[1:, :]

    if qmc:
        estimates = np.empty((n_maturities, len(K), qmc_error_estimators))
        for i in range(qmc_error_estimators):
            if verbose >= 1:
                print(f'Computing estimator {i + 1} of {qmc_error_estimators}')
            estimates[:, :, i], _, _ = cf.eur_MC(S_0=S_0, K=K_mat, T=T_vec, r=r, payoff=payoff,
                                                 samples=get_samples(False if i == 0 else True), implied_vol=False)
        est, stat = cf.MC(estimates)
        if not implied_vol:
            if is_smile:
                est, stat = est[0, :], stat[0, :]
            return est, est - stat, est + stat
        smile, lower, upper = cf.iv_eur(S_0=S_0, K=K_mat, r=r, T=T_vec, price=est, payoff=payoff, stat=stat)
        if is_smile:
            smile, lower, upper = smile[0, :], lower[0, :], upper[0, :]
        return smile, lower, upper
    else:
        return cf.eur_MC(S_0=S_0, K=K_mat, T=T_vec, r=r, samples=get_samples(), payoff=payoff, implied_vol=implied_vol)


def price_geom_asian_call(K, lambda_, rho, nu, theta, V_0, S_0, T, nodes, weights, r, m, N_time, euler=False, qmc=True,
                          qmc_error_estimators=25, verbose=0):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    :param K: Strike prices, assumed to be a numpy array
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param T: Final time/Time of maturity
    :param nodes: The nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param r: Interest rate
    :param m: Number of samples
    :param N_time: Number of time steps used in simulation
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param qmc_error_estimators: Runs the pricing step of the Longstaff-Schwartz algorithm qmc_error_estimators times
        to get an MC estimate for the QMC error
    :param verbose: Determines the number of intermediary results printed to the console
    return: The prices of the call option for the various strike prices in K
    """
    n_batches, m_batch = get_n_batches(N=0, return_times=N_time, m=m)

    def get_sample_batch(rng=None, rv_shift=False):
        return samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, nodes=nodes, weights=weights, rho=rho,
                       S_0=S_0, r=r, m=m_batch, N_time=N_time, sample_paths=True, vol_only=False, stock_only=True,
                       euler=euler, qmc=qmc, rv_shift=rv_shift, rng=rng, verbose=verbose - 1)

    def single_estimator(rv_shift=False):
        estimates = np.empty((len(K), n_batches))
        confidences = np.empty((len(K), n_batches))
        rng = None
        for j in range(n_batches):
            samples_, rng = get_sample_batch(rng=rng, rv_shift=rv_shift)
            estimates[:, j], low, upp = cf.price_geom_asian_call_MC(K=K, samples=samples_)
            confidences[:, j] = np.abs(estimates[:, j] - low)
            if qmc:
                rng.reset()
                rng.fast_forward((j + 1) * m_batch)
        estimate, confidence = cf.MC(estimates)
        if n_batches < 100:
            confidence, _ = cf.MC(confidence) / np.sqrt(n_batches)
        return estimate, estimate - confidence, estimate + confidence

    if qmc:
        estimators = np.empty((len(K), qmc_error_estimators))
        for i in range(qmc_error_estimators):
            if verbose >= 1:
                print(f'Computing estimator {i + 1} of {qmc_error_estimators}')
            estimators[:, i], _, _ = single_estimator(rv_shift=i != 0)
        est, stat = cf.MC(estimators)
        return est, est - stat, est + stat
    else:
        return single_estimator()


def price_avg_vol_call(K, lambda_, nu, theta, V_0, T, nodes, weights, m, N_time, euler=False, qmc=True,
                       qmc_error_estimators=25, verbose=0):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    :param K: Strike prices, assumed to be a numpy array
    :param lambda_: Mean-reversion speed
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param nodes: The nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param m: Number of samples
    :param N_time: Number of time steps used in simulation
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param qmc_error_estimators: Runs the pricing step of the Longstaff-Schwartz algorithm qmc_error_estimators times
        to get an MC estimate for the QMC error
    :param verbose: Determines the number of intermediary results printed to the console
    return: The prices of the call option for the various strike prices in K
    """
    def get_samples(rv_shift=False):
        samples_, rng_ = samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, nodes=nodes, weights=weights,
                                 rho=0., S_0=1., r=0., m=m, N_time=N_time, sample_paths=True, vol_only=True,
                                 stock_only=False, euler=euler, qmc=qmc, rv_shift=rv_shift, verbose=verbose - 1)
        return samples_[0, :, :]

    if qmc:
        estimates = np.empty(qmc_error_estimators)
        for i in range(qmc_error_estimators):
            if verbose >= 1:
                print(f'Computing estimator {i + 1} of {qmc_error_estimators}')
            samples_1 = get_samples(False if i == 0 else True)
            estimates[:, i], _, _ = cf.price_avg_vol_call_MC(K=K, samples=samples_1)
        est, stat = cf.MC(estimates)
        return est, est - stat, est + stat
    else:
        return cf.price_geom_asian_call_MC(K=K, samples=get_samples())


def am_features(x, degree=6, K=0.):
    """
    Computes the features for the pricing of American options under Markovian approximations of rough Heston using the
    Longstaff-Schwartz algorithm.
    :param x: Array of samples of the Markov process. Is of shape (N + 1, m), where N is the dimension of the Markovian
        approximation and m is the number of samples. The array x[0, :] is the array of stock price samples, the array
        x[1:, :] is the array of the N-dimensions Markovian approximation of the volatility process, already multiplied
        with their weights (i.e. w_i V^i_t)
    :param degree: Maximal weighted degree of features polynomials. The degree of the stock price carries weight 1,
        the degree of the total volatility carries weight 2, and the degree of the components of the volatility carries
        weight 3. The last (highest mean-reversion) component of the volatility is not used
    :param K: Strike price. Rough size of the stock price for normalization
    return: The feature vector, of shape (m, dim), where dim is the number of features
    """
    m = x.shape[-1]
    d = x.shape[0]
    normalized_stock = ((x[0, :] - K) / K) if np.abs(K) > 0.01 else x[0, :]
    vol = np.sum(x[1:, :], axis=0)
    vol_factors = x[1:-1, :].T
    if degree == 1:
        dim = 1
    elif degree == 2:
        dim = 3
    else:
        dim = d + 3
    feat = np.empty((m, dim))
    if degree >= 1:
        feat[:, 0] = normalized_stock
    if degree >= 2:
        feat[:, 1] = normalized_stock ** 2
        feat[:, 2] = vol
    if degree >= 3:
        feat[:, 3:5] = normalized_stock[:, None] * feat[:, 1:3]
        feat[:, 5:d + 3] = vol_factors
    current_N = 4
    current_ind = d + 3
    lower_N_stock = 3
    upper_N_stock = d + 3
    lower_N_vol = 2
    upper_N_vol = 3
    next_lower_N_vol = 5
    next_upper_N_vol = d + 3
    lower_N_vol_factors = np.arange(5, d + 3, dtype=int)
    upper_N_vol_factors = d + 3
    while current_N <= degree:
        feat_new = np.empty((m, feat.shape[1] + upper_N_stock - lower_N_stock + upper_N_vol - lower_N_vol
                             + (current_N % 3 == 0)
                             * (upper_N_vol_factors * (d - 2) - np.sum(lower_N_vol_factors, dtype=int))))
        # print(feat_new.shape[1])
        feat_new[:, :current_ind] = feat
        feat = feat_new
        next_ind = current_ind + upper_N_stock - lower_N_stock
        feat[:, current_ind:next_ind] = normalized_stock[:, None] * feat[:, lower_N_stock:upper_N_stock]
        lower_N_stock = current_ind
        current_ind = next_ind
        next_ind = current_ind + upper_N_vol - lower_N_vol
        feat[:, current_ind:next_ind] = vol[:, None] * feat[:, lower_N_vol:upper_N_vol]
        lower_N_vol = next_lower_N_vol
        next_lower_N_vol = current_ind
        current_ind = next_ind
        if current_N % 3 == 0:
            for i in range(d - 2):
                next_ind = current_ind + upper_N_vol_factors - lower_N_vol_factors[i]
                feat[:, current_ind:next_ind] = \
                    vol_factors[:, i:i + 1] * feat[:, lower_N_vol_factors[i]:upper_N_vol_factors]
                lower_N_vol_factors[i] = current_ind
                current_ind = next_ind
            upper_N_vol_factors = current_ind
        upper_N_stock = current_ind
        upper_N_vol = next_upper_N_vol
        next_upper_N_vol = current_ind
        current_N = current_N + 1
    return feat


def price_am(K, lambda_, rho, nu, theta, V_0, S_0, T, nodes, weights, payoff, r, m, N_time, N_dates, feature_degree=6,
             euler=False, qmc=True, qmc_error_estimators=25, verbose=0):
    """
    Gives the price of an American option in the approximated, Markovian rough Heston model.
    :param K: Strike price
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param T: Final time/Time of maturity
    :param nodes: The nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param payoff: The payoff function, either 'call' or 'put' or a function taking as inputs S (samples) and K (strike)
    :param r: Interest rate
    :param m: Number of samples. Uses m for fitting the stopping rule, and another m for pricing
    :param N_time: Number of time steps used in the simulation
    :param N_dates: Number of exercise dates. If None, N_dates = N_time
    :param feature_degree: The degree of the polynomial features used in the regression
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param qmc_error_estimators: Runs the pricing step of the Longstaff-Schwartz algorithm qmc_error_estimators times
        to get an MC estimate for the QMC error
    :param verbose: Determines the number of intermediary results printed to the console
    return: The prices of the call option for the various strike prices in K
    """
    if payoff == 'call':
        def payoff(S):
            return cf.payoff_call(S=S, K=K)
    elif payoff == 'put':
        def payoff(S):
            return cf.payoff_put(S=S, K=K)
    if N_dates is None:
        N_dates = N_time

    def features(x):
        return am_features(x=x, degree=feature_degree, K=K)

    def get_samples(rng_=None, rv_shift=False):
        samples_, rng_ = samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, nodes=nodes, weights=weights,
                                 rho=rho, S_0=S_0, r=r, m=m, N_time=N_time, sample_paths=True, return_times=N_dates,
                                 vol_only=False, stock_only=False, euler=euler, qmc=qmc, rng=rng_, rv_shift=rv_shift)
        samples_ = np.transpose(samples_, (0, 2, 1))
        samples_[1:-1, :, :] = weights[:, None, None] * samples_[2:, :, :]
        samples_ = samples_[:-1, :, :]
        samples_[1:, :, :] = samples_[1:, :, :] - samples_[1:, :, :1]
        return samples_, rng_

    if qmc:
        rng = None
        estimates = np.empty(qmc_error_estimators)
        for i in range(qmc_error_estimators):
            if verbose >= 1:
                print(f'Computing estimator {i + 1} of {qmc_error_estimators}')
            samples_1, rng = get_samples(rng, False if i == 0 else True)
            (_, _), models = cf.price_am(T=T, r=r, samples=samples_1, payoff=payoff, features=features)
            rng.reset()
            rng.fast_forward(m)
            samples_1, rng = get_samples(rng, False if i == 0 else True)
            rng.reset()
            estimates[i], _ = cf.price_am_forward(T=T, r=r, samples=samples_1, payoff=payoff, models=models,
                                                  features=features)
        est, stat = cf.MC(estimates)
    else:
        samples_1, rng = get_samples()
        (_, _), models = cf.price_am(T=T, r=r, samples=samples_1, payoff=payoff, features=features)
        samples_1, rng = get_samples(rng)
        est, stat = cf.price_am_forward(T=T, r=r, samples=samples_1, payoff=payoff, models=models, features=features)
    return est, stat
