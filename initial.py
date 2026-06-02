from test_custom_env2 import custom_visualization


initial_positions = [
    [0.7216, 0.9954, 0.025],
    [0.8987, 0.9827, 0.025],
    [0.4934, 0.1935, 0.025],
    [0.943, 0.1906, 0.025],
    [0.536, 0.9867, 0.025],
    [0.7922, 0.1982, 0.025],
    [0.6422, 0.2069, 0.025],
]

yaws = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]



manual_block_lengths = [0.1, 0.1, 0.1, 0.074, 0.125,0.088, 0.1125]
cliff0_center = 0.3
cliff1_center = 1.0

if __name__ == "__main__":
    custom_visualization(yaws, initial_positions, manual_block_lengths, cliff0_center, cliff1_center)