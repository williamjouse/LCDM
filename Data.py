import numpy as np



#SNe Ia

SN = np.loadtxt('data/lcparam_full_long_zhel.txt', usecols=(1, 2, 4, 5), dtype=np.float64)
zcmb, zhel, mb, dmu = SN[:, 0], SN[:, 1], SN[:, 2], SN[:, 3]

D_stat = np.dot(np.diag(dmu),np.diag(dmu))

C_sys = np.loadtxt('data/sys_full_long.txt')
C_sys = np.reshape(C_sys, (1048, 1048))

C = np.add(D_stat, C_sys)
invSN = np.linalg.inv(C)


#CC
H_data =  np.loadtxt('data/H_z31.txt', dtype=np.float64)

z_H, H_exp, H_err = H_data[:,0], H_data[:,1], H_data[:,2]


#BAO2D

BAO2D = np.loadtxt('data/BAO.txt', dtype=np.float64)

z_theta, theta, theta_err = BAO2D[:,0], BAO2D[:,1]*(np.pi/180.e0), BAO2D[:,2]*(np.pi/180.e0)


CMB_data = np.loadtxt('data/CMB_2018.txt', dtype=np.float64)
X_obs, X_err = np.array([CMB_data[:,0 ]]), np.array([CMB_data[:, 1]])

CMB_c = np.loadtxt('data/CMB_corr_2018.txt', dtype=np.float64)
CMB_c = np.reshape(CMB_c, (3, 3))
CMB_stat = np.dot(np.diag(X_err), np.diag(X_err))
CMB_cov = np.add(CMB_c, CMB_stat)

inv_CMB = np.linalg.inv(CMB_cov)


data = np.loadtxt('data/SDDS-final.txt', dtype=np.float32)
zeff1, DVrd, errDVrd = data[0:2,0], data[0:2,1], data[0:2,2]
zeff2, DMrd, errDMrd = data[2:8,0], data[2:8,1], data[2:8,2]
zeff3, DHrd, errDHrd = data[8:,0], data[8:,1], data[8:,2]

datafs8 = np.loadtxt('data/fs8-new.txt')
zfs8, fs8, sgm_fs8, H_fiducial, DA_fiducial = datafs8[:, 0], datafs8[:, 1], datafs8[:, 2], datafs8[:,3], datafs8[:,4]

datafs8_wigglez = np.loadtxt('data/fs8_wigglez.txt')
zfs8_wigglez, fs8_wigglez, sgm_fs8_wigglez, H_fiducial_wigglez, DA_fiducial_wigglez = datafs8_wigglez[:, 0], \
                            datafs8_wigglez[:, 1], datafs8_wigglez[:, 2], datafs8_wigglez[:,3], datafs8_wigglez[:,4]

fs8_c = np.loadtxt('data/fs8_cov.txt', dtype=np.float64)
fs8_c = np.reshape(fs8_c, (3, 3))
fs8_stat = np.dot(np.diag(sgm_fs8_wigglez), np.diag(sgm_fs8_wigglez))
fs8_cov = np.add(fs8_c, fs8_stat)

inv_fs8 = np.linalg.inv(fs8_cov)


