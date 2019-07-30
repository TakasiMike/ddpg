from matplotlib import pyplot as plt
import numpy as np
R = 8.314462

# Tsonopoulos model

def beta_tson(Tr, omega, Tc, Pc):
    f0 = 0.1445 - (0.330/Tr) - (0.1385/Tr**2) - (0.0121/Tr**3) - (0.000607/Tr**8)
    f1 = 0.0637 + (0.331/Tr*2) - (0.423/Tr**3) - (0.008/Tr**8)
    B_tson = (f0 + omega * f1) * R * Tc / Pc
    return B_tson

# Abbott Model

def betaAb(Tr, omega, Tc, Pc):
    B0 = 0.083 - (0.422/Tr**1.6)
    B1 = 0.139 - (0.172/Tr**4.2)
    B_ab = (B0 + omega * B1) * R * Tc / Pc
    return B_ab


# model = input("Select desired thermodynamical model : ")
# Tr = float(input("Enter Tr :"))
# omega = float(input("Enter omega :"))
# Tc = float(input("Enter Critical Temperature :"))
# Pc = float(input("Enter Critical Pressure : "))
# for i in range(2):
#     model = input("Select desired thermodynamical model : ")
#     Tr = float(input("Enter Tr :"))
#     omega = float(input("Enter omega :"))
#     Tc = float(input("Enter Critical Temperature :"))
#     Pc = float(input("Enter Critical Pressure : "))
#     if model == 'Tsonopoulos':
#         print('Beta factor according to Tsonopoulos model is : ' + str(beta_tson(Tr, omega, Tc, Pc)))
#     elif model == 'Abbott':
#         print('Beta factor according to Abbott model is : ' + str(betaAb(Tr, omega, Tc, Pc)))
#     else:
#         print("This is not a valid model, please select either Tsonopoulos or Abbott model")


model = input("Select desired thermodynamical model (Tsonopoulos or Abbott) : ")

omega = float(input("Enter omega :"))
Tc = float(input("Enter Critical Temperature :"))
Pc = float(input("Enter Critical Pressure : "))
try:
    if model == 'Tsonopoulos':
        for T in range(250, 570, 10):
            Tr = T / Tc
            print(T, (10 ** 6) * beta_tson(Tr, omega, Tc, Pc))
        Temp = np.linspace(0.588, 1.317)
        plt.plot(Temp, (10 ** 6) * beta_tson(Temp, omega, Tc, Pc))
        plt.show()
    elif model == 'Abbott':
        for T in range(250, 570, 10):
            Tr = T / Tc
            print(T,  (10 ** 6) * betaAb(Tr, omega, Tc, Pc))
        Temp = np.linspace(0.588, 1.317)
        plt.plot(Temp, (10 ** 6) * betaAb(Temp, omega, Tc, Pc))
        plt.show()
    else:
        print("This is not a valid model, please select either Tsonopoulos or Abbott model")

    if model == 'Tsonopoulos':
        for Tr in range(250, 560):
            beta_tson(Tr, omega, Tc, Pc)
except:
    print(3)





