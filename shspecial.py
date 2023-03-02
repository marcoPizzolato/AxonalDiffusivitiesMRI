import numpy as np
import scipy.special as ssp

sqrtpi = np.sqrt(np.pi)

def psifun(order,x):
    if order==0:
        return psifun_00(x)
    elif order==2:
        return psifun_02(x)
    elif order==4:
        return psifun_04(x)
    elif order==6:
        return psifun_06(x)
    elif order==8:
        return psifun_08(x)
    elif order==10:
        return psifun_10(x)
    elif order==12:
        return psifun_12(x)
    else:
        raise ValueError('order not supported')



def psifun_12(x):
    
    #sqrtpi = np.sqrt(np.pi)
    if x==0:
        phi_00 = 2
        phi_02 = 2/3
        phi_04 = 2/5
        phi_06 = 2/7
        phi_08 = 2/9
        phi_10 = 2/11
        phi_12 = 2/13
    else:
        x_012 = np.sqrt(x)
        x_032 = x_012*x
        x_052 = x_032*x
        x_072 = x_052*x
        x_092 = x_072*x
        x_112 = x_092*x
        x_132 = x_112*x

        sqrtpierfsqrtx = sqrtpi*ssp.erf(x_012)

        twoexpmxsqrtx = 2*np.exp(-x) * x_012
        x_2 = x*x
        x_3 = x_2*x
        x_4 = x_3*x
        x_5 = x_4*x

        
        phi_00 = sqrtpierfsqrtx / x_012
        phi_02 = (sqrtpierfsqrtx - twoexpmxsqrtx) / (2*x_032)
        phi_04 = (3*sqrtpierfsqrtx - twoexpmxsqrtx * (2*x+3)) / (4*x_052)
        phi_06 = (15*sqrtpierfsqrtx - twoexpmxsqrtx * (4*x_2 + 10*x + 15)) / (8*x_072)
        phi_08 = (105*sqrtpierfsqrtx - twoexpmxsqrtx * (8*x_3 + 28*x_2 + 70*x + 105)) / (16*x_092)
        phi_10 = (945*sqrtpierfsqrtx - twoexpmxsqrtx * (16*x_4 + 72*x_3 + 252*x_2 + 630*x +945)) / (32*x_112)
        phi_12 = (10395*sqrtpierfsqrtx - twoexpmxsqrtx * (32*x_5 + 176*x_4 + 792*x_3 + 2772*x_2 + 6930*x + 10395)) / (64*x_132)

    psi_00 = phi_00
    psi_02 = (3*phi_02 - phi_00) / 2
    psi_04 = (35*phi_04 - 30*phi_02 + 3*phi_00) / 8
    psi_06 = (231*phi_06 - 315*phi_04 + 105*phi_02 - 5*phi_00) / 16
    psi_08 = (6435*phi_08 - 12012*phi_06 + 6930*phi_04 - 1260*phi_02 + 35*phi_00) / 128
    psi_10 = (46189*phi_10 - 109395*phi_08 + 90090*phi_06 - 30030*phi_04 +3465*phi_02 -63*phi_00) / 256
    psi_12 = (676039*phi_12 - 1939938*phi_10 + 2078505*phi_08 - 1021020*phi_06 + 225225*phi_04 - 18018*phi_02 + 231*phi_00) / 1024
    return np.array([psi_00,psi_02,psi_04,psi_06,psi_08,psi_10,psi_12])

def psifun_10(x):
    
    if x==0:
        phi_00 = 2
        phi_02 = 2/3
        phi_04 = 2/5
        phi_06 = 2/7
        phi_08 = 2/9
        phi_10 = 2/11
        
    else:
        #sqrtpi = np.sqrt(np.pi)
        x_012 = np.sqrt(x)
        x_032 = x_012*x
        x_052 = x_032*x
        x_072 = x_052*x
        x_092 = x_072*x
        x_112 = x_092*x

        sqrtpierfsqrtx = sqrtpi*ssp.erf(x_012)

        twoexpmxsqrtx = 2*np.exp(-x) * x_012
        x_2 = x*x
        x_3 = x_2*x
        x_4 = x_3*x

        phi_00 = sqrtpierfsqrtx / x_012
        phi_02 = (sqrtpierfsqrtx - twoexpmxsqrtx) / (2*x_032)
        phi_04 = (3*sqrtpierfsqrtx - twoexpmxsqrtx * (2*x+3)) / (4*x_052)
        phi_06 = (15*sqrtpierfsqrtx - twoexpmxsqrtx * (4*x_2 + 10*x + 15)) / (8*x_072)
        phi_08 = (105*sqrtpierfsqrtx - twoexpmxsqrtx * (8*x_3 + 28*x_2 + 70*x + 105)) / (16*x_092)
        phi_10 = (945*sqrtpierfsqrtx - twoexpmxsqrtx * (16*x_4 + 72*x_3 + 252*x_2 + 630*x +945)) / (32*x_112)
    
    psi_00 = phi_00
    psi_02 = (3*phi_02 - phi_00) / 2
    psi_04 = (35*phi_04 - 30*phi_02 + 3*phi_00) / 8
    psi_06 = (231*phi_06 - 315*phi_04 + 105*phi_02 - 5*phi_00) / 16
    psi_08 = (6435*phi_08 - 12012*phi_06 + 6930*phi_04 - 1260*phi_02 + 35*phi_00) / 128
    psi_10 = (46189*phi_10 - 109395*phi_08 + 90090*phi_06 - 30030*phi_04 +3465*phi_02 -63*phi_00) / 256
    
    return np.array([psi_00,psi_02,psi_04,psi_06,psi_08,psi_10])

def psifun_08(x):
    
    if x==0:
        phi_00 = 2
        phi_02 = 2/3
        phi_04 = 2/5
        phi_06 = 2/7
        phi_08 = 2/9
    else:#sqrtpi = np.sqrt(np.pi)
        x_012 = np.sqrt(x)
        x_032 = x_012*x
        x_052 = x_032*x
        x_072 = x_052*x
        x_092 = x_072*x
        
        sqrtpierfsqrtx = sqrtpi*ssp.erf(x_012)

        twoexpmxsqrtx = 2*np.exp(-x) * x_012
        x_2 = x*x
        x_3 = x_2*x
        
        phi_00 = sqrtpierfsqrtx / x_012
        phi_02 = (sqrtpierfsqrtx - twoexpmxsqrtx) / (2*x_032)
        phi_04 = (3*sqrtpierfsqrtx - twoexpmxsqrtx * (2*x+3)) / (4*x_052)
        phi_06 = (15*sqrtpierfsqrtx - twoexpmxsqrtx * (4*x_2 + 10*x + 15)) / (8*x_072)
        phi_08 = (105*sqrtpierfsqrtx - twoexpmxsqrtx * (8*x_3 + 28*x_2 + 70*x + 105)) / (16*x_092)
    
    psi_00 = phi_00
    psi_02 = (3*phi_02 - phi_00) / 2
    psi_04 = (35*phi_04 - 30*phi_02 + 3*phi_00) / 8
    psi_06 = (231*phi_06 - 315*phi_04 + 105*phi_02 - 5*phi_00) / 16
    psi_08 = (6435*phi_08 - 12012*phi_06 + 6930*phi_04 - 1260*phi_02 + 35*phi_00) / 128
    
    return np.array([psi_00,psi_02,psi_04,psi_06,psi_08])

def psifun_06(x):
    
    if x==0:
        phi_00 = 2
        phi_02 = 2/3
        phi_04 = 2/5
        phi_06 = 2/7
        
    else:#sqrtpi = np.sqrt(np.pi)
        x_012 = np.sqrt(x)
        x_032 = x_012*x
        x_052 = x_032*x
        x_072 = x_052*x
        
        sqrtpierfsqrtx = sqrtpi*ssp.erf(x_012)

        twoexpmxsqrtx = 2*np.exp(-x) * x_012
        x_2 = x*x
        
        phi_00 = sqrtpierfsqrtx / x_012
        phi_02 = (sqrtpierfsqrtx - twoexpmxsqrtx) / (2*x_032)
        phi_04 = (3*sqrtpierfsqrtx - twoexpmxsqrtx * (2*x+3)) / (4*x_052)
        phi_06 = (15*sqrtpierfsqrtx - twoexpmxsqrtx * (4*x_2 + 10*x + 15)) / (8*x_072)
    
    psi_00 = phi_00
    psi_02 = (3*phi_02 - phi_00) / 2
    psi_04 = (35*phi_04 - 30*phi_02 + 3*phi_00) / 8
    psi_06 = (231*phi_06 - 315*phi_04 + 105*phi_02 - 5*phi_00) / 16
    
    return np.array([psi_00,psi_02,psi_04,psi_06])

def psifun_04(x):
    
    if x==0:
        phi_00 = 2
        phi_02 = 2/3
        phi_04 = 2/5
        
    else:
        #sqrtpi = np.sqrt(np.pi)
        x_012 = np.sqrt(x)
        x_032 = x_012*x
        x_052 = x_032*x
        
        sqrtpierfsqrtx = sqrtpi*ssp.erf(x_012)

        twoexpmxsqrtx = 2*np.exp(-x) * x_012
        
        phi_00 = sqrtpierfsqrtx / x_012
        phi_02 = (sqrtpierfsqrtx - twoexpmxsqrtx) / (2*x_032)
        phi_04 = (3*sqrtpierfsqrtx - twoexpmxsqrtx * (2*x+3)) / (4*x_052)
    
    psi_00 = phi_00
    psi_02 = (3*phi_02 - phi_00) / 2
    psi_04 = (35*phi_04 - 30*phi_02 + 3*phi_00) / 8
    
    return np.array([psi_00,psi_02,psi_04])

def psifun_02(x):
    
    if x==0:
        phi_00 = 2
        phi_02 = 2/3
        
    else:#sqrtpi = np.sqrt(np.pi)
        x_012 = np.sqrt(x)
        x_032 = x_012*x
        
        sqrtpierfsqrtx = sqrtpi*ssp.erf(x_012)

        twoexpmxsqrtx = 2*np.exp(-x) * x_012
        
        phi_00 = sqrtpierfsqrtx / x_012
        phi_02 = (sqrtpierfsqrtx - twoexpmxsqrtx) / (2*x_032)
    
    psi_00 = phi_00
    psi_02 = (3*phi_02 - phi_00) / 2
    
    return np.array([psi_00,psi_02])

def psifun_00(x):
    
    if x==0:
        phi_00 = 2
    
    else:#sqrtpi = np.sqrt(np.pi)
        x_012 = np.sqrt(x)
        
        sqrtpierfsqrtx = sqrtpi*ssp.erf(x_012)

        twoexpmxsqrtx = 2*np.exp(-x) * x_012
        
        phi_00 = sqrtpierfsqrtx / x_012
    
    psi_00 = phi_00
    
    return np.array([psi_00])
