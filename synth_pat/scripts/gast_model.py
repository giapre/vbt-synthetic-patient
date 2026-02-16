import jax.numpy as jp
import collections  

# Model implementation

Theta = collections.namedtuple(
    typename='Theta',
    field_names='C, k, Delta, v_r, v_theta, g, E, I, tau_u, b, kappa, tau_s, J, Js, we,')

default_theta = Theta(
    C=100.0, k=0.7, Delta=0.5, v_r=-60.0, v_theta=-40.0, g=1.0, E=0.0, I=40.0, 
    tau_u=33.33, b=-2.0, kappa=100.0, tau_s=6.0, Js=.0, J=15.0, we=0.0, 
)

def dfun(y, cy, p: Theta):
    "Adaptive QIF model with dopamine modulation."

    r, v, u, s = y
    c_exc = cy
    C, k, Delta, v_r, v_theta, g, E, I, tau_u, b, kappa, tau_s, J, Js, *_ = p

    dr = ((Delta * k**2 * (v - v_r)) / (jp.pi * C) + r * (k * (2 * v - v_r - v_theta) - g*s)) / C
    dv = (k * v * (v - v_r - v_theta) - jp.pi * C * r * (Delta + jp.pi * C * r/ k) + k * v_r * v_theta - u + I + g * s * (E - v)) / C
    du = (b * (v - v_r) -u) / tau_u + kappa * r
    ds = - s / tau_s + Js * c_exc + J * r

    return jp.array([dr, dv, du, ds])

def net(y, p):
    "Canonical form for network of dopa nodes."
    Ce, node_params = p
    r = y[0]
    c_exc = node_params.we * Ce @ r
  
    return dfun(y, (c_exc), node_params)

def stay_positive(y, _):
    # at, set are JAX function used for immutable updates to an array
    # if where<0 is true, set the value to 0, conversely it leaves the original value 
    # in this way r, is never negative
    y = y.at[0].set( jp.where(y[0]<0, 0, y[0]) ) #r

    return y

# Model implementation with dopamine

dopa_Theta = collections.namedtuple(
    typename='dopa_Theta',
    field_names='C, k, Delta, v_r, v_theta, Bd, ga, gg, Ea, Eg, I, tau_u, b, kappa, tau_sa, tau_sg, Ja, Jg, Jsa, Jsg, Jdopa, Vmax, Km, tau_Dp, Rd, Sd, Z, tau_Md, we, wi, wd')

dopa_default_theta = dopa_Theta(
    C=100.0, k=0.7, Delta=0.5, v_r=-60.0, v_theta=-40.0, Bd=1., ga=1.0, gg=1., Ea=0.0, Eg=-80., I=46.5, 
    tau_u=33.33, b=-2.0, kappa=100.0, tau_sa=6.0, tau_sg=6., Ja=13., Jg=0., Jsa=13., Jsg=15., 
    Jdopa=100000.0, Vmax=1300.0, Km=150.0, Rd=1., Sd=-10.0, Z=.5, tau_Dp=500.0, tau_Md=1000.0, 
    we=1e-2, wi=1e-2, wd=1e-2,
)

def dopa_dfun(y, cy, p: dopa_Theta):
    "Adaptive QIF model with dopamine modulation."

    r, v, u, sa, sg, Dp, Md = y
    c_exc, c_inh, c_dopa = cy
    C, k, Delta, v_r, v_theta, Bd, ga, gg, Ea, Eg, I, tau_u, b, kappa, tau_sa, tau_sg, Ja, Jg, Jsa, Jsg, Jdopa, Vmax, Km, tau_Dp, Rd, Sd, Z, tau_Md, *_ = p

    dr = ((Delta * k**2 * (v - v_r)) / (jp.pi * C) + r * (k * (2 * v - v_r - v_theta) - (Bd + Md) * ga * sa - gg * sg)) / C
    dv = (k * v * (v - v_r - v_theta) - jp.pi * C * r * (Delta + jp.pi * C * r/ k) + k * v_r * v_theta - u + I + (Bd + Md) * ga * sa * (Ea - v) + gg * sg * (Eg - v)) / C
    du = (b * (v - v_r) -u) / tau_u + kappa * r
    dsa = - sa / tau_sa + Jsa * c_exc + Ja * r
    dsg = - sg / tau_sg + Jsg * c_inh + Jg * r
    dDp = (Jdopa * c_dopa - Vmax * Dp / (Km + Dp)) / tau_Dp
    dMd = (-Md + Rd / (1 + jp.exp(Sd * jp.log((Dp+Z))))) / tau_Md

    return jp.array([dr, dv, du, dsa, dsg, dDp, dMd])

def dopa_net(y, p):
    "Canonical form for network of dopa nodes."
    Ce, Ci, Cd, node_params = p
    r = y[0]
    c_exc = node_params.we * Ce @ r
    c_inh = node_params.wi * Ci @ r
    c_dopa = node_params.wd * Cd @ r

    return dopa_dfun(y, (c_exc, c_inh, c_dopa), node_params)

def dopa_stay_positive(y, _):
    # at, set are JAX function used for immutable updates to an array
    # if where<0 is true, set the value to 0, conversely it leaves the original value 
    # in this way r, Dp and Md are never negative
    y = y.at[0].set( jp.where(y[0]<0, 0, y[0]) ) #r
    y = y.at[5].set( jp.where(y[5]<0, 0, y[5]) ) #Dp
    y = y.at[6].set( jp.where(y[6]<0, 0, y[6]) ) #Md

    return y

# New model implementation with simplified neuromodulators dynamics

d1d2sero_Theta = collections.namedtuple(
    typename='d1d2sero_Theta',
    field_names='C, k, Delta, v_r, v_theta, Bd, Bs, ga, gg, Ea, Eg, I, tau_u, b, kappa, tau_sa, tau_sg, Ja, Jg, Jsa, Jsg, m_D1, R_D1, m_D2, R_D2, m_sero, R_sero, tau_Mdopa, tau_Msero, we, wi, wd, ws, sigma_V, sigma_u')

d1d2sero_default_theta = d1d2sero_Theta(
    C=100.0, k=0.7, Delta=0.5, v_r=-60.0, v_theta=-40.0, Bd=1., Bs=1., ga=1.0, gg=1., Ea=0.0, Eg=-80., I=46.5, 
    tau_u=33.33, b=-2.0, kappa=100.0, tau_sa=6.0, tau_sg=6., Ja=13., Jg=0., Jsa=13., Jsg=15., 
    m_D1=1., R_D1=1., m_D2=1., R_D2=1., m_sero=1., R_sero=1., tau_Mdopa=1000.0, tau_Msero=1000.0,
    we=1e-2, wi=1e-2, wd=1e-2, ws=1e-2, sigma_V=1.0, sigma_u=1.0,
)

def d1d2sero_dfun(y, cy, p: d1d2sero_Theta):
    "Adaptive QIF model with dopamine modulation."

    r, v, u, sa, sg, Md, Ms = y
    c_exc, c_inh, c_dopa, c_sero = cy
    C, k, Delta, v_r, v_theta, Bd, Bs, ga, gg, Ea, Eg, I, tau_u, b, kappa, tau_sa, tau_sg, Ja, Jg, Jsa, Jsg, m_D1, R_D1, m_D2, R_D2, m_sero, R_sero, tau_Mdopa, tau_Msero, *_ = p

    dr = ((Delta * k**2 * (v - v_r)) / (jp.pi * C) + r * (k * (2 * v - v_r - v_theta) - (Bd + Md) * (Bs + Ms) * ga * sa - gg * sg)) / C
    dv = (k * v * (v - v_r - v_theta) - jp.pi * C * r * (Delta + jp.pi * C * r/ k) + k * v_r * v_theta - u + I + (Bd + Md) * (Bs + Ms) * ga * sa * (Ea - v) + gg * sg * (Eg - v)) / C
    du = (b * (v - v_r) -u) / tau_u + kappa * r
    dsa = - sa / tau_sa + Jsa * c_exc + Ja * r
    dsg = - sg / tau_sg + Jsg * c_inh + Jg * r
    dMd = (- Md + (m_D1 * R_D1 - m_D2 * R_D2) * c_dopa) / tau_Mdopa
    dMs = (- Ms + (m_sero * R_sero) * c_sero) / tau_Msero

    return jp.array([dr, dv, du, dsa, dsg, dMd, dMs])

# New model implementation with sigmoidal dose-response neuromodulators dynamics

sigm_d1d2sero_Theta = collections.namedtuple(
    typename='sigm_d1d2sero_Theta',
    field_names='C, k, Delta, v_r, v_theta, Bd, Bs, ga, gg, Ea, Eg, I, tau_u, b, kappa, tau_sa, tau_sg, Ja, Jg, Jsa, Jsg, Jdopa, Jsero, Vmax_dopa, Vmax_sero, Km_dopa, Km_sero, tau_Dp, tau_sero, Rd1, Rd2, Rs, Sd1, Sd2, Ss, Zd1, Zd2, Zs, tau_Md1, tau_Md2, tau_Ms, we, wi, wd, ws, sigma_V, sigma_u')

sigm_d1d2sero_default_theta = sigm_d1d2sero_Theta(
    C=100.0, k=0.7, Delta=0.5, v_r=-60.0, v_theta=-40.0, Bd=1.0, Bs=1.0,
    ga=1.0, gg=1.0, Ea=0.0, Eg=-80.0, I=46.5, tau_u=33.33, b=-2.0, kappa=100.0, tau_sa=6.0, tau_sg=6.0, Ja=13.0, Jg=0., Jsa=13.0, Jsg=15.0, 
    Jdopa=100000.0, Jsero=100000.0, Vmax_dopa=1300.0, Vmax_sero=90.7, Km_dopa=150.0, Km_sero=6.7, tau_Dp=500.0, tau_sero=500.0, 
    Rd1=1., Rd2=1., Rs=1., Sd1=-10.0, Sd2=-10.0, Ss=-40.0, Zd1=0.5, Zd2=1., Zs=.25, tau_Md1=1000, tau_Md2=1000, tau_Ms=1000,  
    we=1e-2, wi=1., wd=1e-2, ws=1e-2, sigma_V=0., sigma_u=0.,
)

def sigm_d1d2sero_dfun(y, cy, p: sigm_d1d2sero_Theta):
    "Adaptive QIF model with dopamine modulation."

    r, v, u, sa, sg, Dp, Se, Md1, Md2, Ms = y
    c_exc, c_inh, c_dopa, c_sero = cy
    C, k, Delta, v_r, v_theta, Bd, Bs, ga, gg, Ea, Eg, I, tau_u, b, kappa, tau_sa, tau_sg, Ja, Jg, Jsa, Jsg, Jdopa, Jsero, Vmax_dopa, Vmax_sero, Km_dopa, Km_sero, tau_Dp, tau_sero, Rd1, Rd2, Rs, Sd1, Sd2, Ss, Zd1, Zd2, Zs, tau_Md1, tau_Md2, tau_Ms, *_ = p

    dr = ((Delta * k**2 * (v - v_r)) / (jp.pi * C) + r * (k * (2 * v - v_r - v_theta) - (Bd + Md1 - Md2 + Ms) * ga * sa - gg * sg)) / C
    dv = (k * v * (v - v_r - v_theta) - jp.pi * C * r * (Delta + jp.pi * C * r/ k) + k * v_r * v_theta - u + I + (Bd + Md1 - Md2 + Ms) * ga * sa * (Ea - v) + gg * sg * (Eg - v)) / C
    du = (b * (v - v_r) -u) / tau_u + kappa * r
    dsa = - sa / tau_sa + Jsa * c_exc + Ja * r
    dsg = - sg / tau_sg + Jsg * c_inh + Jg * r
    dDp = (Jdopa * c_dopa - Vmax_dopa * Dp / (Km_dopa + Dp)) / tau_Dp
    dSe = (Jsero * c_sero - Vmax_sero * Se / (Km_sero + Se)) / tau_sero
    dMd1 = (- Md1 + (Rd1 / (1 + jp.exp(Sd1 * jp.log(Dp + Zd1))))) / tau_Md1
    dMd2 = (- Md2 + (Rd2 / (1 + jp.exp(Sd2 * jp.log(Dp + Zd2))))) / tau_Md2
    dMs = (- Ms + (Rs / (1 + jp.exp(Ss * jp.log(Se + Zs))))) / tau_Ms

    return jp.array([dr, dv, du, dsa, dsg, dDp, dSe, dMd1, dMd2, dMs])
