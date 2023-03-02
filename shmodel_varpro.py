import numpy as np
from shspecial import psifun
from dipy_dependencies import real_sph_harm
from dipy_dependencies import cart2sphere
from scipy.optimize import differential_evolution, minimize
import multiprocessing as mp
from time import time



_directions724 = np.load('./data/directions724.npy')

def sh_matrix(sh_order, vecs):
    r""" Compute Spherical Harmonics (SH) matrix M
    Parameters
    ----------
    sh_order : int, even
        Truncation order of the SH.
    vecs : array, Nx3
        array unit directions [X,Y,Z].

    Returns
    --------
    M : array,
        The base of the SH sampled according to vecs.
    """

    r, theta, phi = cart2sphere(vecs[:, 0], vecs[:, 1], vecs[:, 2])
    theta[np.isnan(theta)] = 0

    n_c = (sh_order + 1) * (sh_order + 2) // 2
    shm = real_sph_harm

    M = np.zeros((vecs.shape[0], n_c))
    counter = 0
    for l in range(0, sh_order + 1, 2):
        for m in range(-l, l + 1):
            M[:, counter] = shm(m, l, theta, phi)
            counter += 1
    return M

def compute_lm_ordervector(sh_order):
    '''
    Given a spherical harmonics order in the modified convention (i.e. like in
    Descoteaux et al., 2006), returns the list (array) of the L and M order of
    the coefficients that will be used for the representation in SH.
    '''
    n_c = (sh_order + 1) * (sh_order + 2) // 2
    LM = np.zeros((n_c,2),dtype=int)
    counter = 0
    for kl, l in enumerate(range(0, sh_order + 1, 2)):
        #print 'KL ' + str(l)
        for km, m in enumerate(range(-l, l + 1)):
            #print  'km ' + str(m)
            LM[counter,0] = int(l)
            LM[counter,1] = int(m)
            counter = counter+1
    return LM


def _laplace_beltrami(sh_order):
    "Returns the Laplace-Beltrami regularisation matrix for SH basis"
    n_c = (sh_order + 1)*(sh_order + 2)//2
    diagL = np.zeros(n_c)
    counter = 0
    for l in range(0, sh_order + 1, 2):
        for m in range(-l, l + 1):
            diagL[counter] = (l * (l + 1)) ** 2
            counter += 1

    return np.diag(diagL)

def _eye(sh_order):
    "Returns the Laplace-Beltrami regularisation matrix for SH basis"
    n_c = (sh_order + 1)*(sh_order + 2)//2
    diagL = np.zeros(n_c)
    counter = 0
    for l in range(0, sh_order + 1, 2):
        for m in range(-l, l + 1):
            diagL[counter] = 1
            counter += 1

    return np.diag(diagL)



def _laplace_beltrami_nosphmean(sh_order,num_shells):
    "Returns the Laplace-Beltrami regularisation matrix for SH basis"
    n_c = (sh_order + 1)*(sh_order + 2)//2
    diagL = np.zeros(n_c+num_shells)
    counter = 0
    for l in range(0, sh_order + 1, 2):
        for m in range(-l, l + 1):
            diagL[counter] = (l * (l + 1)) ** 2
            counter += 1
    diag_cropped = diagL[1:]
    #diag_flipped = diag_cropped[::-1]
    return np.diag(diag_cropped)

def _eye_nosphmean(sh_order,num_shells):
    "Returns the Laplace-Beltrami regularisation matrix for SH basis"
    n_c = (sh_order + 1)*(sh_order + 2)//2
    diagL = np.zeros(n_c+num_shells)
    counter = 0
    for l in range(0, sh_order + 1, 2):
        for m in range(-l, l + 1):
            diagL[counter] = 1
            counter += 1
    diag_cropped = diagL[1:]
    #diag_flipped = diag_cropped[::-1]
    return np.diag(diag_cropped)


def create_sh_multipliers(b_reference,other_bvalues,D_parallel,D_perp,sh_order):
    multipliers = []

    x_reference = b_reference*(D_parallel-D_perp)
    x_n_list    = other_bvalues*(D_parallel-D_perp)

    psifun_reference = psifun(sh_order,x_reference)
    for k,x_n in enumerate(x_n_list):
        multipliers.append(np.exp(-other_bvalues[k]*D_perp)*psifun(sh_order,x_n)/ ( psifun_reference*np.exp(-b_reference*D_perp)))#[::-1]
    return multipliers

'''INCLUDING SPHERICAL MEAN'''

def extented_sh_matrix(sh_order_list,list_of_bvecs,b_reference_idx):
    n_rows = 0
    sh_order_max = int(np.max(np.array(sh_order_list)))
    for vecs in list_of_bvecs:
        n_rows = n_rows + vecs.shape[0]
    n_cols =  ((sh_order_max + 1) * (sh_order_max + 2) // 2)
    
    sh_extended_matrix = np.zeros((n_rows,n_cols))
    L_order_extended_matrix = np.zeros((n_rows,n_cols))*np.nan
    previous_row_number = 0
    num_shells = len(list_of_bvecs)
    for k,vecs in enumerate(list_of_bvecs):
        sh_order = sh_order_list[k]
        shmtrx = sh_matrix(sh_order, vecs)
        Lmatrix = np.ones_like(shmtrx)
        LM = compute_lm_ordervector(sh_order)
        Lmatrix = Lmatrix*LM[:,0]
        shmtrx_nrows,shmtrx_ncols = shmtrx.shape
        sh_extended_matrix[previous_row_number:previous_row_number+shmtrx_nrows,0:shmtrx_ncols] = shmtrx#shmtrx[:,::-1][:,0:-1]
        L_order_extended_matrix[previous_row_number:previous_row_number+shmtrx_nrows,0:shmtrx_ncols] = Lmatrix
        previous_row_number = previous_row_number+shmtrx_nrows
    
    L_order_indexs = []
    lower_limit = 0#list_of_bvecs[0].shape[0]
    
    for k,vecs in enumerate(list_of_bvecs):
        n_rows_block = vecs.shape[0]
        upper_limit = lower_limit+n_rows_block
        #print(lower_limit,upper_limit)
        L_order_indexs_block = []

        sh_order = sh_order_list[k]
        LM = compute_lm_ordervector(sh_order)
        for l in np.unique(LM[:,0]):
            if l>=0:
                #print(l)
                whereres = np.where(L_order_extended_matrix==l)
                idxs = (whereres[0] < upper_limit) & (whereres[0] >= lower_limit)
                rows = whereres[0][idxs]
                cols = whereres[1][idxs]
                L_order_indexs_block.append(tuple((rows,cols)))
        lower_limit = upper_limit

        if k!=b_reference_idx:
            L_order_indexs.append(L_order_indexs_block)

    return sh_extended_matrix,L_order_extended_matrix,L_order_indexs

def multiply_extented_sh_matrix(sh_extended_matrix,L_order_indexs,multipliers):
    '''
    multiplies by the corresponding factor all of the elements of the matrix having the same SH order l
    '''
    sh_extended_matrix_copy = sh_extended_matrix.copy()
    for block_idxs,block_multipliers in zip(L_order_indexs,multipliers):
        for idxs,mult_values in zip(block_idxs,block_multipliers):
            sh_extended_matrix_copy[idxs] = sh_extended_matrix_copy[idxs]*mult_values
    return sh_extended_matrix_copy


def tensor_sh_model_design_matrix(b_reference,other_bvalues,D_parallel,D_perp,sh_order,sh_extended_matrix,L_order_indexs):
    multipliers = create_sh_multipliers(b_reference,other_bvalues,D_parallel,D_perp,sh_order)
    #print('inside loop',freesphmean)
    scaled_sh_extended_matrix = multiply_extented_sh_matrix(sh_extended_matrix,L_order_indexs,multipliers)
    return scaled_sh_extended_matrix

def tensor_sh_model_varpro(shells,b_reference,other_bvalues,D_parallel,D_perp,sh_order,sh_extended_matrix,L_order_indexs,L,positivity_mtrx_L_order_indexes=None,lambda_tik=0,lb_matrix=0):
    #lambda_tik=0.001
    M = tensor_sh_model_design_matrix(b_reference,other_bvalues,D_parallel,D_perp,sh_order,sh_extended_matrix,L_order_indexs)
    # pseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M)+lambda_tik*np.eye(M.shape[1])), M.T)
    pseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M)+lambda_tik*lb_matrix), M.T)
    # pseudoInv = np.linalg.pinv(M, rcond=0.01)#rcond=1e-15)
    return np.dot(M,np.dot(pseudoInv,shells))

def tensor_sh_objfun_varpro(x,shells,b_reference,other_bvalues,sh_order,sh_extended_matrix,L_order_indexs,L,positivity_mtrx_L_order_indexes=None,lambda_tik=0,lb_matrix=0):
    D_parallel = x[0]
    D_perp     = x[1]

    recon = tensor_sh_model_varpro(shells[0],b_reference,other_bvalues,D_parallel,D_perp,sh_order,sh_extended_matrix,L_order_indexs,L,positivity_mtrx_L_order_indexes,lambda_tik,lb_matrix)
    
    obj_value = 0
    cnt = 0

    for voxel_signal in shells:
        obj_value += np.sum((recon-voxel_signal)**2)
        cnt += 1
    obj_value = obj_value/cnt

    return obj_value



'''NO SPH MEAN'''

def extented_sh_matrix_sphmeanfree(sh_order_list,list_of_bvecs,b_reference_idx):
    n_rows = 0
    sh_order_max = int(np.max(np.array(sh_order_list)))
    for vecs in list_of_bvecs:
        n_rows = n_rows + vecs.shape[0]
    n_cols =  ((sh_order_max + 1) * (sh_order_max + 2) // 2) + len(list_of_bvecs)-1
    
    sh_extended_matrix = np.zeros((n_rows,n_cols))
    L_order_extended_matrix = np.zeros((n_rows,n_cols))*np.nan
    previous_row_number = 0
    num_shells = len(list_of_bvecs)
    for k,vecs in enumerate(list_of_bvecs):
        sh_order = sh_order_list[k]
        shmtrx = sh_matrix(sh_order, vecs)
        Lmatrix = np.ones_like(shmtrx)
        LM = compute_lm_ordervector(sh_order)
        Lmatrix = Lmatrix*LM[:,0]
        shmtrx_nrows,shmtrx_ncols = shmtrx.shape
        sh_extended_matrix[previous_row_number:previous_row_number+shmtrx_nrows,0:shmtrx_ncols-1] = shmtrx[:,1:]#shmtrx[:,::-1][:,0:-1]
        L_order_extended_matrix[previous_row_number:previous_row_number+shmtrx_nrows,0:shmtrx_ncols-1] = Lmatrix[:,1:]

        idx = n_cols-num_shells+k
        sh_extended_matrix[previous_row_number:previous_row_number+shmtrx_nrows,idx] =  shmtrx[:,0]
        L_order_extended_matrix[previous_row_number:previous_row_number+shmtrx_nrows,idx] =  Lmatrix[:,0]

        previous_row_number = previous_row_number+shmtrx_nrows
    
    
    L_order_indexs = []
    lower_limit = 0#list_of_bvecs[0].shape[0]
    
    for k,vecs in enumerate(list_of_bvecs):
        n_rows_block = vecs.shape[0]
        upper_limit = lower_limit+n_rows_block
        #print(lower_limit,upper_limit)
        L_order_indexs_block = []

        sh_order = sh_order_list[k]
        LM = compute_lm_ordervector(sh_order)
        for l in np.unique(LM[:,0]):
            if l>0:
                #print(l)
                whereres = np.where(L_order_extended_matrix==l)
                idxs = (whereres[0] < upper_limit) & (whereres[0] >= lower_limit)
                rows = whereres[0][idxs]
                cols = whereres[1][idxs]
                L_order_indexs_block.append(tuple((rows,cols)))
        lower_limit = upper_limit

        if k!=b_reference_idx:
            L_order_indexs.append(L_order_indexs_block)

    return sh_extended_matrix,L_order_extended_matrix,L_order_indexs


def multiply_extented_sh_matrix_sphmeanfree(sh_extended_matrix,L_order_indexs,multipliers):
    '''
    multiplies by the corresponding factor all of the elements of the matrix having the same SH order l
    '''
    sh_extended_matrix_copy = sh_extended_matrix.copy()
    for block_idxs,block_multipliers in zip(L_order_indexs,multipliers):
        for idxs,mult_values in zip(block_idxs,block_multipliers[1:]):
            sh_extended_matrix_copy[idxs] = sh_extended_matrix_copy[idxs]*mult_values
    return sh_extended_matrix_copy

def tensor_sh_model_design_matrix_sphmeanfree(b_reference,other_bvalues,D_parallel,D_perp,sh_order,sh_extended_matrix,L_order_indexs):
    multipliers = create_sh_multipliers(b_reference,other_bvalues,D_parallel,D_perp,sh_order)
    #print('inside loop',freesphmean)
    scaled_sh_extended_matrix = multiply_extented_sh_matrix_sphmeanfree(sh_extended_matrix,L_order_indexs,multipliers)
    return scaled_sh_extended_matrix

def tensor_sh_model_varpro_sphmeanfree(shells,b_reference,other_bvalues,D_parallel,D_perp,sh_order,sh_extended_matrix,L_order_indexs,L,positivity_mtrx_L_order_indexes=None,lambda_tik=0,lb_matrix=0):
    #lambda_tik=0.001
    M = tensor_sh_model_design_matrix_sphmeanfree(b_reference,other_bvalues,D_parallel,D_perp,sh_order,sh_extended_matrix,L_order_indexs)#[:,:-(int(len(other_bvalues)+1))]
    # pseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M)+lambda_tik*np.eye(M.shape[1])), M.T)
    pseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M)+lambda_tik*lb_matrix), M.T)
    # pseudoInv = np.linalg.pinv(M, rcond=0.01)#rcond=1e-15)
    return np.dot(M,np.dot(pseudoInv,shells))

def tensor_sh_objfun_varpro_sphmeanfree(x,shells,b_reference,other_bvalues,sh_order,sh_extended_matrix,L_order_indexs,L,positivity_mtrx_L_order_indexes=None,lambda_tik=0,lb_matrix=0):
    D_parallel = x[0]
    D_perp     = x[1]

    recon = tensor_sh_model_varpro_sphmeanfree(shells[0],b_reference,other_bvalues,D_parallel,D_perp,sh_order,sh_extended_matrix,L_order_indexs,L,positivity_mtrx_L_order_indexes,lambda_tik,lb_matrix)
    
    obj_value = 0
    cnt = 0

    for voxel_signal in shells:
        obj_value += np.sum((recon-voxel_signal)**2)
        cnt += 1
    obj_value = obj_value/cnt

    return obj_value



def estimate_tensor_sh_worker_varpro(tuple_item):
    start = time()
    
    shells                  = tuple_item[0]
    b_reference             = tuple_item[1]
    other_bvalues           = tuple_item[2]
    sh_order                = tuple_item[3]
    sh_extended_matrix      = tuple_item[4]
    L_order_indexs          = tuple_item[5]
    L                       = tuple_item[6]
    lambda_tik              = tuple_item[7]
    bnds                    = tuple_item[8]
    positivity_mtrx_L_order_indexes = tuple_item[9]
    lb_matrix               = tuple_item[10]
    use_sph_mean            = tuple_item[11]
    tot                     = tuple_item[12]
    x                       = tuple_item[13]
    n_workers               = tuple_item[14]

    if use_sph_mean:
        the_fun = tensor_sh_objfun_varpro
    else:
        the_fun = tensor_sh_objfun_varpro_sphmeanfree
    #print('start worker ',x)
    # result_0 = differential_evolution(func=the_fun,
    #                             args=(shells,b_reference,other_bvalues,sh_order,sh_extended_matrix,L_order_indexs,L,positivity_mtrx_L_order_indexes,lambda_tik,lb_matrix,),
    #                             bounds=bnds)

    ###############x0 = np.array([2e-9,0.005e-9])
    x0 = np.array([(bnds[0][0]+bnds[0][1])/2,(bnds[1][0]+bnds[1][1])/2])
    result_0 = minimize(the_fun, x0, method='L-BFGS-B', bounds=bnds,
                options={'disp': None, 'maxcor': 20, 'ftol': 2.220446049250313e-13, 'gtol': 1e-11, 'eps': 1e-13, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None},
                args=(shells,b_reference,other_bvalues,sh_order,sh_extended_matrix,L_order_indexs,L,positivity_mtrx_L_order_indexes,lambda_tik,lb_matrix,),)
    
    
    D_parallel = result_0.x[0]
    D_perp = result_0.x[1]
    #print(M)
    prev_time = time()-start
    verbose=False
    if x % 20 ==0:
        verbose=True
    strt = manage_time_print((x+1),tot,prev_time/n_workers,verbose=verbose)

    return D_parallel,D_perp


def fit_tensor_sh_all_data_varpro(data_rcs,dirs_xyz,bvals,bvals_unique,sh_order_list,n_workers,mask,D_par_low,D_par_high,D_perp_low,D_perp_high,constrain_sph_mean=False,lambda_tik=1e-001,positivity=False,use_sph_mean=True,neighboorhood_fit=False,neighboorhood_radius=1,neighboorhood_size=2,reg='LB'):
    '''
    Accepts data where the spherical mean has been subtracted in a previous step
    '''
    if not use_sph_mean:
        print('ATTENTION: you should have subtracted the spherical mean from the data before passing it as input')
    
    print('SH ORDER LIST only works with equal SH order per shell so far e.g [12,12]')
    
    nR,nC,nS,ndirs = data_rcs.shape
    d_parallel = np.zeros((nR,nC,nS))
    d_perp = np.zeros((nR,nC,nS))
    
    sh_order_max = int(np.max(np.array(sh_order_list)))
    unique_sh_order = True
    for el in sh_order_list:
        unique_sh_order = (el==sh_order_max) and (unique_sh_order)
    if unique_sh_order:
        b_reference_idx = np.argmin(bvals_unique)
    else:
        b_reference_idx = np.argmax(np.array(sh_order_list))


    b_reference = bvals_unique[b_reference_idx]
    other_bvalues = np.zeros(len(bvals_unique)-1)
    cnt=0
    for k in range(len(bvals_unique)):
        if k!=b_reference_idx:
            other_bvalues[cnt] = bvals_unique[k]
    
    list_of_bvecs = []
    list_of_bvecs_positivity = []
    
    if constrain_sph_mean:
        sphmeans = np.zeros((nR,nC,nS,len(bvals_unique)))

    
    cnt = 0
    for k,b in enumerate(bvals_unique):
        idxs = bvals==b
        list_of_bvecs.append(dirs_xyz[idxs,:])
        list_of_bvecs_positivity.append(_directions724)
        if constrain_sph_mean:
            sphmeans[:,:,:,k] = np.mean(data_rcs[:,:,:,idxs],axis=-1)
        

    n_shells = len(bvals_unique)
    shm = real_sph_harm
    
    if use_sph_mean:
        sh_extended_matrix, L_order_extended_matrix, L_order_indexs = extented_sh_matrix(sh_order_list,list_of_bvecs,b_reference_idx)    
        if reg=='LB':
            lb_matrix = _laplace_beltrami(sh_order_max)
        elif reg=='eye':
            lb_matrix = _eye(sh_order_max)
        else:
            raise ValueError("wrong reg specified")
    else:
        sh_extended_matrix, L_order_extended_matrix, L_order_indexs = extented_sh_matrix_sphmeanfree(sh_order_list,list_of_bvecs,b_reference_idx)    
        if reg=='LB':
            lb_matrix = _laplace_beltrami_nosphmean(sh_order_max,n_shells)
        elif reg=='eye':
            lb_matrix = _eye_nosphmean(sh_order_max,n_shells)
        else:
            raise ValueError("wrong reg specified")

    if positivity:
        print('positivity==True, however feature not yet implemented, no positivity constraints will be enforced')
        positivity_mtrx_L_order_indexes = None
    else:
        positivity_mtrx_L_order_indexes = None

    if constrain_sph_mean:
        print('constrain_sph_mean==True, however feature not yet implemented, ignoring..')
    
    L = np.zeros((sh_extended_matrix.shape[0],sh_extended_matrix.shape[0]))

    
    #coeffs = np.zeros((nR,nC,nS,sh_extended_matrix.shape[1]))
    
    if neighboorhood_fit:
        print('Building neighborhood')
        print('Attention: this feature will collect data from the voxel neighborhood and optimize it at once: the feature is not tested and is under design: set neighboorhood_fit=False for safer results')

    list_data_idxs = []
    tot = np.sum(mask.astype(int))
    cnt = 0
    for r in range(nR):
        for c in range(nC):
            for s in range(nS):
                if mask[r,c,s]==1:
                    

                    data2give = data_rcs[r,c,s,:]
                    
                    # neighborhood_signals = [data2give]
                    # signal_distances = [0.]
                    
                    neighborhood_signals = []
                    signal_distances = []

                    if neighboorhood_fit:
                        therange = list(range(-neighboorhood_radius,neighboorhood_radius+1,1))
                        for rr in therange:
                            for cc in therange:
                                for ss in therange:#[0]:
                                    if (r+rr>=0) & (r+rr<nR) & (c+cc>=0) & (c+cc<nC) & (s+ss>=0) & (s+ss<nS):
                                        if mask[r+rr,c+cc,s+ss]==1:

                                            data2give_tmp = data_rcs[r+rr,c+cc,s+ss,:]
                                            
                                            neighborhood_signals.append(data2give_tmp)
                                            #signal_distances.append(np.mean((np.array(data2give_tmp)-np.array(data2give))**2))
                                            signal_distances.append((np.mean(data2give_tmp)-np.mean(data2give))**2)
                        
                        

                       
                        sorted_neighborhood_signals = [data2give]
                        idxs_sort = np.argsort(np.array(signal_distances))
                        #idxs_sort=idxs_sort[::-1]
                        for nghbhd_idx in range(len(neighborhood_signals)):
                            sorted_neighborhood_signals.append(neighborhood_signals[idxs_sort[nghbhd_idx]])
                            
                        if neighboorhood_size<=len(sorted_neighborhood_signals):
                             sorted_neighborhood_signals=sorted_neighborhood_signals[0:neighboorhood_size+1]
                    
                    else:
                        sorted_neighborhood_signals = [data2give]


                    # print(neighborhood_signals)

                    #print('data2give',len(data2give))
                    bnds = ((D_par_low[r,c,s],D_par_high[r,c,s]),(D_perp_low[r,c,s],D_perp_high[r,c,s]),)
                    list_data_idxs.append((sorted_neighborhood_signals,b_reference,other_bvalues,sh_order_max,sh_extended_matrix,L_order_indexs,L,lambda_tik,bnds,positivity_mtrx_L_order_indexes,lb_matrix,use_sph_mean,tot,cnt,n_workers))
                    cnt = cnt + 1
    
    print('Fit...')

    # cnt=0
    # for r in range(nR):
    #     for c in range(nC):
    #         for s in range(nS):
    #             if mask[r,c,s]==1:
    #                 estimate_tensor_sh_worker_varpro(list_data_idxs[cnt])
    #                 cnt+=1


    pool = mp.Pool(processes=n_workers)
    results = [pool.apply_async(estimate_tensor_sh_worker_varpro, args=(list_data_idxs[x],)) for x in range(cnt)]#int(np.sum(mask_cpmg))
    pool.close()
    pool.join()

    cnt=0
    for r in range(nR):
        for c in range(nC):
            for s in range(nS):
                if mask[r,c,s]==1:
                    result_item = results[cnt]
                    cnt = cnt + 1
                    fit_result = result_item.get()
                    d_parallel[r,c,s]    = fit_result[0]
                    d_perp[r,c,s]     = fit_result[1]
                    #coeffs[r,c,s,:]     = fit_result[2]

    return d_parallel,d_perp#,coeffs










# ################################################################################
# ################################################################################



def manage_time_print(k,tot,prev_time,verbose=True):

    if k>0:
       if int((tot-k)*prev_time) < 60:
           toappend = str(int((tot-k)*prev_time )) + ' seconds'
       else:
           toappend = str(int((tot-k)*prev_time /60. )) + ' minutes'
       if verbose==True:
           print(str(k) + ' out of ' +str(tot) + ' complete' + ' expected time ' + toappend, end="\r")
    else:
       if verbose==True:
           print(str(k) + ' out of ' +str(tot) + ' complete', end="\r")
    start = time()
    return start

