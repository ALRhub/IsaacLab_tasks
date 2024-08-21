import matplotlib.pyplot as plt


def plot_joint_trajectories(traj_isaaclab, traj_fg):
    num_plots = len(traj_isaaclab[0])

    # Create a figure and subplots
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 15))

    # Iterate over each index in the sublists
    for i in range(num_plots):
        # Extract the i-th elements from each sublist in both lists
        elements_isaaclab = [sublist[i] for sublist in traj_isaaclab]
        elements_fg = [sublist[i] for sublist in traj_fg]

        # Plot both lists on the same subplot
        axs[i].plot(elements_isaaclab, label="IsaacLab")
        axs[i].plot(elements_fg, label="FancyGym")
        axs[i].set_title(f"Plot for joint {i}")
        axs[i].legend()
        axs[i].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.show()
