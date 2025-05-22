from CFD import CFD

is_forest = int(input('Enter 1 for forest and 0 for tree: '))
rom = int(input('Enter the order of ROM (0, 1): '))
num_branches = int(input('Enter the number of branches: '))
obj = CFD()
obj.set_parameters(num_branches=num_branches)
obj.set_assumptions(convex = True)

if is_forest == 0:
    obj.create_directory(rom, num_branches, is_forest)
    obj.implicit()
    obj.tree_build()
    if rom == 0:
        obj.export_tree_0d_files()
        obj.run_tree_0d_simulation()
        obj.plot_0d_results_to_3d()
    elif rom == 1:
        obj.export_tree_0d_files() # saves the model files
        obj.export_tree_1d_files()
        obj.run_tree_1d_simulation()
else:
    num_networks = int(input('Enter the number of networks: '))
    trees_per_network = list(map(int, input("Enter number of trees in each network separated by space: ").split()))
    obj.create_directory(rom, num_branches, is_forest)
    obj.implicit()
    obj.forest_build(num_networks, trees_per_network)
    obj.export_forest_0d_files(num_networks=num_networks)