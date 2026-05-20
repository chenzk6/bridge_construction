from test_custom_env2 import custom_visualization


initial_positions = [
    [0.7887, 0.2486, 0.025],
    [0.6399, 0.2476, 0.025],
    [0.4886, 0.2244, 0.025],
    [0.847, 0.9623, 0.025],
    [0.7454, 0.9705, 0.025],
    [0.5123, 0.9451, 0.025],
    [0.6321, 0.9495, 0.025],
]

yaws = [
    -3.121934,
    -3.090989,
    0.025307,
    0.005252,
    3.078153,
    -3.137221,
    0.009976,
]



manual_block_lengths = [0.1, 0.1, 0.1, 0.074, 0.125,0.088, 0.1125]
cliff0_center = 0.3
cliff1_center = 1.05

if __name__ == "__main__":
    custom_visualization(yaws, initial_positions, manual_block_lengths, cliff0_center, cliff1_center)