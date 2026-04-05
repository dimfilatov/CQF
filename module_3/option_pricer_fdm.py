import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Set max row and columns to 300 and 100
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 100)

def option_price_with_finite_difference_method(K, T, r, sigma, call, NAS, american=False):
    # set up grid parameters
    S_max = 2 * K
    S_min = 0
    ds = S_max / NAS
    dt = 0.9/sigma**2 / NAS**2 # stability condition for explicit method
    NTS = int(T/dt) + 1 # number of time steps
    dt = T / NTS # adjust dt to fit exactly into T
    s = np.arange(0,(NAS+1)*ds, ds) # stock price grid
    t = np.arange(dt * NTS, 0, -dt) # time grid
    # initialize option price grid
    grid = np.zeros((NAS+1, NTS ))
    # set terminal condition at maturity
    if call:
        grid[ :, -1] = np.maximum(s-K, 0)
    else:
        grid[ :, -1] = np.maximum(K-s, 0)
    
    intrinsic_values = grid[ :, -1].copy()
    
    # fill the grid
    for i in range(NTS-2, -1, -1):
        for j in range(1, NAS):
            delta = (grid[j+1, i+1] - grid[j-1, i+1])/(2*ds)
            gamma = (grid[j+1, i+1] - 2*grid[j, i+1] + grid[j-1, i+1])/(ds**2)
            theta = -0.5*sigma**2 * s[j]**2 * gamma - r*s[j]*delta + r*grid[j, i+1]

            # calculate the option price at current grid point
            grid[j, i] = grid[j, i+1] - dt * theta

        # set boundary condition for S = 0
        grid[0, i] = 0 if call else grid[0, i+1] * np.exp(-r*dt)
        
        # for s = s_max we know that gamma = 0
        grid[-1, i] = 2*grid[-2, i] - grid[-3, i]

        if american:
            grid[:, i] = np.maximum(intrinsic_values, grid[:, i])
        
    grid = np.around(grid, 2)
    
    return grid, s, t, intrinsic_values

def plot_option_value(grid, s, t, intrinsic_values):  
    grid_df = pd.DataFrame(grid, index=s, columns=t)
    print(grid)

    option_value_2D = pd.DataFrame({"Stock":s, "Payoff": intrinsic_values, "Option_value": grid_df.iloc[:,0]})
    print(option_value_2D)

    # --- 2D Plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(s, intrinsic_values, label='Payoff', linestyle='--')
    ax.plot(s, grid_df.iloc[:, 0], label='Option Value')
    ax.set_title('Payoff & Option Value')
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 3D Surface Plot ---
    X, Y = np.meshgrid(t, s)
    Z = grid

    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.85)
    ax3d.set_title('Option Values by Explicit FDM')
    ax3d.set_xlabel('Time to Maturity')
    ax3d.set_ylabel('Stock Price')
    ax3d.set_zlabel('Option Value')
    plt.tight_layout()
    plt.show()

def bilinear_interpolation(s, ttm, grid):

    # find closest rows and columns
    col_l = grid.columns[grid.columns<ttm][-1]
    col_h = grid.columns[grid.columns>ttm][0]
    row_l = grid.index[grid.index<s][-1]
    row_h = grid.index[grid.index>s][0]

    # define points and ares for interpolation
    V_l_l = grid.loc[row_l, col_l]
    V_l_h = grid.loc[row_l, col_h] 
    V_h_l = grid.loc[row_h, col_l]
    V_h_h = grid.loc[row_h, col_h]

    A_l_l = (col_h - ttm) * (row_h - s)
    A_l_h = (col_h - ttm) * (s - row_l)
    A_h_l = (ttm - col_l) * (row_h - s)
    A_h_h = (ttm - col_l) * (s - row_l)

    A = A_l_l + A_l_h + A_h_l + A_h_h

    # calculate interpolated value
    return V_l_l * A_l_l / A + V_l_h * A_l_h / A + V_h_l * A_h_l / A + V_h_h * A_h_h / A



if __name__ == "__main__":
    grid, s, t, intrinsic_values = option_price_with_finite_difference_method(K=100, T=1, r=0.05, sigma=0.2, call=True, NAS=100, american=False)
    plot_option_value(grid, s, t, intrinsic_values)
    bilinear_value = bilinear_interpolation(s=105, ttm=0.5, grid=pd.DataFrame(grid, index=s, columns=t))
    print(f"Bilinear Interpolation Value at S=105, TTM=0.5: {bilinear_value:.2f}")