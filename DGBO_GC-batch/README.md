This code is implemented according to paper "Scalable and Parallel Deep Bayesian Optimization on Attributed graphs".
DGBO/PDGBO methods can deal with attributed graphs. It prevents the cubical complexity of the GPs by adopting a deep graph neural network to surrogate black-box functions, and can scale linearly with the number of observations. Applications include molecular discovery and urban road network design.

This code is implemented for the DGBO with Graph Convolution (DGBO_{GC}).
Folders and their corresponding functions are listed as follows:

    datasets/ :the datasets used in our experiments
    display/ :some functions to display results
    gcn/ :the code for graph convolution
    rdkit_preprocessing/ :required code to preprocess the raw data, e.g., SMILES strings of molecules
    results/ : used to store the results
    
If you want to run this code, you should ensure that you have installed the following packages:

    tensorFlow
    spicy
    pickle
    numpy
    emcee
    networkx

    #for graphNet
    graph_nets
    sonnet

    #optional packages:
    sklearn
    bayes_opt ##via "pip install bayesian-optimization" to install. The website of this package is https://github.com/fmfn/BayesianOptimization

    (Please see the "dependency_packages.md" in their respective folders to find the detailed versions of these packages.)

After you installed all dependency packages, choose and go into "DGBO_GC-batch" or "DGBO_GN-batch" folder, and then you can run DGBO  with the default setting as:

    $$ python DGBO.py --run=True

, or you can see the help message by running as:

    $$ python DGBO.py -h

Note: If you try the zinc dataset, you should run “genConvMolFeatures.py” in “rdkit_preprocessing/” to convert SMILES strings to attributed graphs including xxx-attr.pkl, xxx-graph.pkl, and xxx-label.pkl.
