import math
import numpy as np
from scipy.optimize import nnls, least_squares, brentq
from scipy.signal import argrelextrema

def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu)/sig, 2.)/2) * 1./(np.sqrt(2.*np.pi)*sig)

def gaussian_unnormalized(x, mu, sig):
    return np.exp(-np.power((x - mu)/sig, 2.)/2)

def broaden_density(D, omega, delta):
    if(delta == 0):
        return np.array(D)
    n_omega = omega.size
    d_omega = (omega[-1]-omega[0])/(n_omega-1)
    D_broadend = np.zeros(n_omega)
    for i in range(n_omega):
        D_broadend[i] = np.sum(gaussian(omega, omega[i], delta)*D*d_omega)
    return D_broadend

def simple_truncation(D, omega, cut):
    omega= omega[D>cut]
    D = D[D>cut]
    return D, omega

def peak_truncation_asymmetric(omega, D, cut, center = None):
    #adapted from Alex's code
    if(center is None):
        center = omega[np.argmax(D[:len(D)//2])]
    D_neg=D[omega<center]
    omega_neg=omega[omega<center]
    # cuts the first minimum
    try:
        first=np.max(omega_neg[D_neg<cut])
        D_neg=D_neg[omega_neg>first]
        omega_neg=omega_neg[omega_neg>first]
    except:
        omega_neg=omega_neg[D_neg>cut]
        D_neg=D_neg[D_neg>cut]
    D_pos=D[omega>=center]
    omega_pos=omega[omega>=center]
    omega_pos=omega_pos[D_pos>cut]
    D_pos=D_pos[D_pos>cut]
    D=np.append(D_neg,D_pos)
    omega=np.append(omega_neg,omega_pos)
    omega=np.array(omega[D>=0], dtype=omega.dtype)
    D=np.array(D[D>=0], dtype=D.dtype)
    return omega,D

def peak_truncation(omega, D, cut, center = None):
    if(center is None):
        center = omega[np.argmax(D)]


    D_neg=D[omega<center]
    omega_neg=omega[omega<center]
    # cuts the first minimum
    try:
        first=np.max(omega_neg[D_neg<cut])
        D_neg=D_neg[omega_neg>first]
        omega_neg=omega_neg[omega_neg>first]
    except:
        omega_neg=omega_neg[D_neg>cut]
        D_neg=D_neg[D_neg>cut]
    D_pos=D[omega>=center]
    omega_pos=omega[omega>=center]
    try:
        first=np.min(omega_pos[D_pos<cut])
        D_pos=D_pos[omega_pos<first]
        omega_pos=omega_pos[omega_pos<first]
    except:
        omega_pos=omega_pos[D_pos>cut]
        D_pos=D_pos[D_pos>cut]

    omega_pos=omega_pos[D_pos>cut]
    D_pos=D_pos[D_pos>cut]
    D=np.append(D_neg,D_pos)
    omega=np.append(omega_neg,omega_pos)
    return omega,D    

def quantile_truncation(D, trunc_left, trunc_right):
    D = np.array(D)
    total_weight = np.sum(D)
    start = 0
    weight = 0
    while(weight<=trunc_left*total_weight):
        weight += D[start]
        start += 1
    D[:start] = 0
    D[start-1] = weight-trunc_left*total_weight
    end = D.size-1
    weight = 0
    while(weight<=trunc_right*total_weight):
        weight += D[end]
        end -= 1
    D[end+1:] = 0
    D[end+1] = weight-trunc_right*total_weight
    D = D*(total_weight/np.sum(D))
    return D

def filter_timeseries(G, t, delta):
    if(delta==0):
        return np.array(G)
    return G*gaussian_unnormalized(t, 0, 1.0/delta)

def fourier_transform(G, t, omega):        
    n_t = t.size
    dt = (t[-1]-t[0])/(n_t-1)    
    t_mesh, omega_mesh = np.meshgrid(t, omega)
    # Direct Fourier Transform Matrix
    F = np.exp(1j * t_mesh * omega_mesh)*dt/(2*math.pi)
    D = np.dot(F, G).real

    return D

def symmetrize_timeseries(G, t):
    t2 = np.zeros(2*t.size+1, dtype=t.dtype)
    t2[:t.size] = -1*np.flip(t)
    t2[t.size] = 0.0
    t2[t.size+1:] = t

    G2 = np.zeros(2*t.size+1, dtype=G.dtype)
    G2[:t.size] = np.conjugate(np.flip(G))
    G2[t.size] = 1.0
    G2[t.size+1:] = G

    return G2, t2


def adjust_moments(D, omega, d_omega, exact_mean, exact_width):
  D = D/np.sum(D*d_omega)
  mean = np.sum(D*d_omega*omega)
  width = math.sqrt(np.sum(D*d_omega*(omega-mean)**2))
  omega = omega-mean
  omega = omega*exact_width/width
  omega = omega+exact_mean
  return D, omega, d_omega*exact_width/width

def integrate_density(D, omega, d_omega, beta, delta):
    W = np.sum(np.exp(-beta*omega)*D*d_omega)

    E = np.sum(np.exp(-beta*omega)*D*omega*d_omega)/W
    E_sq = np.sum(np.exp(-beta*omega)*D*omega**2*d_omega)/W

    # corect for broadening
    C = beta*delta**2
    W = W/np.exp(C*beta/2)
    E = E+C        
    E_sq = E_sq + 2*C*E - C**2 - delta**2

    return W, E, E_sq

    
##################################################################
def vstack_real_imag(M):
    return np.vstack([M.real, M.imag])

def hstack_real_imag(M):
    return np.hstack([M.real, M.imag])

def get_inverse_fourier_matrix(t, omega, weights):
    if(t is None):
        return np.zeros((0, omega.size))
    else:
        t_mesh, omega_mesh = np.meshgrid(t, omega)
        IF = np.exp(-1j * t_mesh * omega_mesh).T
        if(weights is None):
            return vstack_real_imag(IF)
        else:
            return vstack_real_imag(IF)*np.reshape(hstack_real_imag(weights), (-1, 1))

def get_inverse_fourier_vector(G_t, weights):
    if(G_t is None):
        return np.zeros((0))

    if(weights is None):
        return hstack_real_imag(G_t)
    else:
        return hstack_real_imag(G_t)*hstack_real_imag(weights)

def get_laplace_matrix(beta, omega, weights):
    if(beta is None):
        return np.zeros((0, omega.size))
    else:
        beta_mesh, omega_mesh = np.meshgrid(beta, omega)
        L = np.exp(-1 * beta_mesh * omega_mesh).T
        if(weights is None):
            return L
        else:
            return L*np.reshape(weights, (-1, 1))

def get_laplace_vector(p, weights):
    if(p is None):
        return np.zeros((0))

    if(weights is None):
        return p
    else:
        return p*weights

def get_moments_matrix(mean, width, omega, weight):
    n_omega = omega.size
    if(mean is None and width is None):
        M = np.zeros((1, n_omega))
        M[0, :] = 1.0
    elif(mean is not None and width is None):
        M = np.zeros((2, n_omega))
        M[0, :] = 1.0
        M[1, :] = omega
    else:
        M = np.zeros((3, n_omega))
        M[0, :] = 1.0
        M[1, :] = omega
        M[2, :] = (omega-mean)**2
        
    if(weight is None):
        return M
    else:
        return M*weight

def get_moments_vector(mean, width, weight):
    if(mean is None and width is None):
        v = np.array([1.0])        
    elif(mean is not None and width is None):
        v = np.array([1.0, mean])
    else:
        v =  np.array([1.0, mean, width**2])

    if(weight is None):
        return v
    else:
        return v*weight

def get_mat_vec(omega, t, G_t, G_t_weights, beta, p_beta, p_beta_weights,  mean, width, moments_weight):
    A = np.vstack([ get_moments_matrix(mean, width, omega, moments_weight), 
                    get_inverse_fourier_matrix(t, omega, G_t_weights), 
                    get_laplace_matrix(beta, omega, p_beta_weights),
                    ])
    b = np.hstack([ get_moments_vector(mean, width, moments_weight), 
                    get_inverse_fourier_vector(G_t, G_t_weights), 
                    get_laplace_vector(p_beta, p_beta_weights),
                    ])
    return A, b

def non_negative_least_squares( omega, 
                                t = None, G_t = None, G_t_weights = None , 
                                beta = None, p_beta = None, p_beta_weights = None,  
                                mean = None, width = None, moments_weight=10, 
                                max_nnls_iters=10000):                            
    A, b = get_mat_vec(omega, t, G_t, G_t_weights, beta, p_beta, p_beta_weights,  mean, width, moments_weight)
    D, _ = nnls(A, b, max_nnls_iters)

    d_omega = (omega[-1]-omega[0])/(omega.size-1)    
    return D/d_omega

def non_negative_least_squares_filtered( omega, expected_fit_squared,
                                t = None, G_t = None, G_t_weights = None , 
                                beta = None, p_beta = None, p_beta_weights = None,  
                                mean = None, width = None, moments_weight=10, 
                                max_nnls_iters=10000):                            
    A, b = get_mat_vec(omega, t, G_t, G_t_weights, beta, p_beta, p_beta_weights,  mean, width, moments_weight)
    D, _ = nnls(A, b, max_nnls_iters)
    d_omega = (omega[-1]-omega[0])/(omega.size-1)
    if(mean is None and width is None):
        moments_count = 1
    elif(mean is not None and width is None):
        moments_count = 2
    else:
        moments_count = 3

    def discrepency(log_q):
        q = 10**log_q
        truncated_D = quantile_truncation(D, q/2, q/2)
        return np.linalg.norm(np.dot(A[moments_count:, :], truncated_D)-b[moments_count:])**2-expected_fit_squared


    if(discrepency(-16)>0 or discrepency(-1e-3)<0):
        return D/d_omega, 0

    log_q = brentq(discrepency, -16, -1e-3)
    q = 10**log_q
    truncated_D = quantile_truncation(D, q/2, q/2)
    return truncated_D/d_omega, q

def sparse_least_squares(omega, k,
                        t = None, G_t = None, G_t_weights = None , 
                        beta = None, p_beta = None, p_beta_weights = None,  
                        mean = None, width = None, moments_weight=10,
                        nnls_init = True, max_nnls_iters=10000, nnls_broadening = 3,
                        enforce_non_negativity = False, max_nfev=10000, loss='linear', f_scale=1, method='trf',gtol=1e-8, ftol=1e-8, xtol=1e-8):

    def fit_func(x):
        D = x[:k]
        omega = x[k:]
        A, b = get_mat_vec(omega, t, G_t, G_t_weights, beta, p_beta, p_beta_weights,  mean, width, moments_weight)
        return np.dot(A, D) - b

    d_omega = (omega[-1]-omega[0])/(omega.size-1)

    if(nnls_init):
        # Initilize using top k peaks of non-negative least squares
        D = non_negative_least_squares(omega, t, G_t, G_t_weights, beta, p_beta, p_beta_weights,  mean, width, moments_weight, max_nnls_iters)
        D = broaden_density(D, omega, nnls_broadening*d_omega)
        extrema = argrelextrema(D, np.greater)[0]
        D_extrema = D[extrema]
        sortedIndices = extrema[np.argsort(-D_extrema)]

        if(k <= len(extrema)):
            omega0 = omega[sortedIndices[:k]]
            D0 = D[sortedIndices[:k]]
        else:
            #split smallest peak into l smaller peaks
            l = k-len(extrema)+1
            omega0 = np.append(omega[sortedIndices[:-1]], [omega[sortedIndices[-1]]]*l)
            D0 = np.append(D[sortedIndices[:-1]], [D[sortedIndices[-1]]/l]*l)

        D0 = D0/np.sum(D0)
    else:
        omega0 = k*[0]
        D0 = k*[1.0/k]
        
    x0 = np.hstack([D0, omega0])

    if(enforce_non_negativity):
        result = least_squares(fit_func, x0, bounds=(np.hstack([[0]*k, [-np.inf]*k]), [np.inf]*(2*k)), max_nfev=max_nfev, loss=loss, f_scale=f_scale, method=method, gtol=gtol, ftol=ftol, xtol=xtol)
    else:
        result = least_squares(fit_func, x0, max_nfev=max_nfev, loss=loss, f_scale=f_scale, method=method,gtol=gtol, ftol=ftol, xtol=xtol)
    D = result.x[:k]
    omega = result.x[k:]

    return D, omega