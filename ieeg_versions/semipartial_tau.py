## Calculate semi-partial (only controlling covariates for one variable) Kendall-B Tau

# define semipartial tau functions
# one covariate
def sptau_onevar(x,y,z):
  # compute semi-partial kendall tau between x and y|z
  # uses the approach described here: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/partktau.htm, https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190238
  # see also https://www.e-sciencecentral.org/upload/csam/pdf/csam-22-665.pdf eqn. 2.2 for an analogous equation in the case of Spearman correlation
  # x is neural, y is model layer and z is covariate model layer
   xy = stats.kendalltau(x,y)
   xz = stats.kendalltau(x,z)
   yz = stats.kendalltau(y,z)
   if any(np.isnan([xy[0],xz[0],yz[0]])):
     print('I found a nan. One of the input vectors might be constant.')
     out = float("nan")
   else:
     out = (xy[0]-xz[0]*yz[0])/np.sqrt(1-yz[0]**2)
   return out

# multiple covariates, adapted from scipy function
def conDisPairs(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size:
        raise ValueError("All inputs must be of the same size, "
                         "found x-size %s and y-size %s" % (x.size, y.size))
    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype('int64', copy=False)
        cnt = cnt[cnt > 1]
        return ((cnt * (cnt - 1) // 2).sum(),
            (cnt * (cnt - 1.) * (cnt - 2)).sum(),
            (cnt * (cnt - 1.) * (2*cnt + 5)).sum())
    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)
    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)
    dis = _kendall_dis(x, y)  # discordant pairs
    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.where(obs)[0]).astype('int64', copy=False)
    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, x0, x1 = count_rank_tie(x)     # ties in x, stats
    ytie, y0, y1 = count_rank_tie(y)     # ties in y, stats
    tot = (size * (size - 1)) // 2
    ######
    #con = tot - (dis + (xtie - ntie) + (ytie - ntie) + ntie)
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    con = con_minus_dis + dis
    tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)
    #con = tot - dis - ntie
    ####
    # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #               = con + dis + xtie + ytie - ntie
    #con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    #SD = (tot - xtie - ytie + ntie - 2 * dis) / (tot - ntie)
    return xtie, ytie, dis, con_minus_dis, tot, tau

#ntiesx, ntiesy, ndisc, diff, tot, tau = conDisPairs(x,y)

def mykendalltau(x,y):
  ntiesx, ntiesy, ndisc, diff, tot, tau = conDisPairs(x,y)
  #out = (diff)/np.sqrt((tot+ntiesx)*(tot+ntiesy))
  return out

def mykendallcov(x,y):
  ntiesx, ntiesy, ndisc, diff, tot, tau = conDisPairs(x,y)
  out = diff
  return out

def ptau_fun(x,y,l):
  if len(l) == 1:
  	allvars = [x,y,l]
  else:
  	allvars = [x,y,*l]
  n = len(allvars)
  n_covars = len(l)
  # calculate Kendall covariance matrix
  V = np.zeros([n,n])
  for i,var1 in enumerate(allvars):
    for j,var2 in enumerate(allvars):
      V[i,j] = mykendallcov(var1,var2)
  Vi = np.linalg.pinv(V, hermitian=True)  # Inverse covariance matrix
  Vi_diag = Vi.diagonal()
  D = np.diag(np.sqrt(1 / Vi_diag))
  ptau = -1 * (D @ Vi @ D)
  ptau[np.diag_indices_from(ptau)] = 1
  return V,Vi,ptau

#V,Vi,ptau= ptau_fun(x,y,l)

def sptau_fun(x,y,l):
  V,Vi,ptau = ptau_fun(x,y,l)
  sptau = ptau[0,1]/np.sqrt(V[0,0])/np.sqrt(Vi[0,0]-(Vi[0,1]*Vi[1,0])/Vi[1,1])
  return sptau

#sptau = sptau_fun(x,y,l)
