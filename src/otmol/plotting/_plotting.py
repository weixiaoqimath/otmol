import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np

ATOMIC_NAME = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br"
}

ATOMIC_COLOR = {
    "H": "silver",
    "C": "black",
    "N": "blue",
    "O": "red",
    "F": "green",
    "P": "orange",
    "S": "yellow",
    "Cl": "limegreen",
    "Br": "salmon"
}

ATOMIC_SIZE = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "Ne": 0.69,
}

ATOMIC_PROPERTIES = {
    "H": {"en": 2.20, "vdw": 1.10, "cov": 0.32},   # Hydrogen (H)
    "C": {"en": 2.55, "vdw": 1.70, "cov": 0.76},   # Carbon (C)
    "N": {"en": 3.04, "vdw": 1.55, "cov": 0.71},   # Nitrogen (N)
    "O": {"en": 3.44, "vdw": 1.52, "cov": 0.66},   # Oxygen (O)
    "F": {"en": 3.98, "vdw": 1.47, "cov": 0.64},   # Fluorine (F)
    "P": {"en": 2.19, "vdw": 1.80, "cov": 1.06},  # Phosphorus (P)
    "S": {"en": 2.58, "vdw": 1.80, "cov": 1.02},  # Sulfur (S)
    "Cl": {"en": 3.16, "vdw": 1.75, "cov": 0.99},  # Chlorine (Cl)
    "Br": {"en": 2.96, "vdw": 1.85, "cov": 1.14},  # Bromine (Br)
}

def plot_molecule(
    X,
    T,
    B,
    coloring = 'uniform',
    sizing = 'uniform',
    point_size = 150,
    edge_size = 3
):
    """
    Simple molecular plotting
    """
    n = X.shape[0]
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n):
        if coloring == 'uniform':
            color = 'red'
        elif coloring == 'type':
            color = ATOMIC_COLOR[T[i]]
        if sizing == 'uniform':
            pt_size = point_size
        elif sizing == 'type':
            pt_size = point_size * ATOMIC_SIZE[T[i]]
        ax.scatter(X[i,0], X[i,1], X[i,2], c=color, s=pt_size)
    for i in range(n-1):
        for j in range(i+1, n):
            if B[i,j] != 0:
                ax.plot([X[i,0], X[j,0]],[X[i,1], X[j,1]],[X[i,2], X[j,2]],
                    color='grey', linewidth=edge_size)
    return

def plot_alignment(
    X1: np.ndarray,
    X2: np.ndarray,
    B1: np.ndarray,
    B2: np.ndarray,
    P: np.ndarray,
    weight_cutoff: float = 1e-5
    ) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1[:,0], X1[:,1], X1[:,2], c='r', s=150)
    ax.scatter(X2[:,0], X2[:,1], X2[:,2], c='b', s=150)
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    for i in range(n1):
        for j in range(n2):
            if P[i,j] > weight_cutoff:
                ax.plot([X1[i,0],X2[j,0]],[X1[i,1],X2[j,1]],[X1[i,2],X2[j,2]], c='black', linewidth=3)
    for i in range(n1):
        for j in range(n1):
            if B1[i,j] != 0:
                ax.plot([X1[i,0],X1[j,0]],[X1[i,1],X1[j,1]],[X1[i,2],X1[j,2]], c='gray', linewidth=3)
    for i in range(n2):
        for j in range(n2):
            if B2[i,j] != 0:
                ax.plot([X2[i,0],X2[j,0]],[X2[i,1],X2[j,1]],[X2[i,2],X2[j,2]], c='gray', linewidth=3)
    return


def interactive_molecule_plot(
    X: np.ndarray,
    T: np.ndarray,
    B: np.ndarray,
    name: str,
    save: bool = False,
    show_labels: bool = True,
    ) -> None:
    fig = go.Figure()
    # Add scatter points for structure A with different sizes for different atoms
    fig.add_trace(go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers+text' if show_labels else 'markers',  # Add text mode
        text=T if show_labels else None,  # Add element labels
        textposition="top center",  # Position the text above the points
        marker=dict(
            size=[ATOMIC_SIZE[label]*20 for label in T], 
            color='red',
            symbol='circle'
        ),
        name=name
    ))

    # Add bonds for A
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            if B[i,j] == 1:  # if there's a bond
                fig.add_trace(go.Scatter3d(
                    x=[X[i,0], X[j,0]],
                    y=[X[i,1], X[j,1]],
                    z=[X[i,2], X[j,2]],
                    mode='lines',
                    line=dict(
                        color='red',
                        width=2
                    ),
                    showlegend=False
                ))

    # Update layout
    fig.update_layout(
        title='name',
        scene=dict(
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis = dict(visible=False),
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
        ),
        showlegend=True,
    )

    # Save the figure as HTML
    if save:
        fig.write_html("{}.html".format(name,))

    config = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            #'filename': 'custom_image',
            'height': 500,
            'width': 700,
            'scale':2 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }
    # Show the figure
    fig.show(config=config)


def interactive_alignment_plot(
    X_A: np.ndarray,
    X_B: np.ndarray,
    T_A: np.ndarray,
    T_B: np.ndarray,
    B_A: np.ndarray,
    B_B: np.ndarray,
    assignment: np.ndarray = None,
    nameA: str = 'A', 
    nameB: str = 'B', 
    save: bool = False,
    show_labels: bool = True,
    show_atom_indices: bool = False,
    only_A_bonds: bool = False,
    ) -> None:
    """Plot the alignment of two structures in 3D in an interactive plot.

    Parameters
    ----------
    X_A : np.ndarray
        The coordinates of the atoms in structure A.
    X_B : np.ndarray
        The coordinates of the atoms in structure B.
    T_A : np.ndarray
        The atom labels of structure A.
    T_B : np.ndarray
        The atom labels of structure B.
    B_A : np.ndarray
        The bond matrix of structure A.
    B_B : np.ndarray
        The bond matrix of structure B.
    assignment: np.ndarray
        1d array. The assignment of atoms in A to atoms in B. 
        The i-th atom in A is assigned to the assignment[i]-th atom in B.
    nameA : str
        The name of structure A.
    nameB : str
        The name of structure B.
    save : bool, optional
        Whether to save the figure as an HTML file.
    only_A_bonds: bool, optional
        If True, plot the bonds in A that are not in B in orange.
    """
    fig = go.Figure()
    # Add scatter points for structure A with different sizes for different atoms
    fig.add_trace(go.Scatter3d(
        x=X_A[:, 0],
        y=X_A[:, 1],
        z=X_A[:, 2],
        mode='markers+text' if show_labels or show_atom_indices else 'markers',  # Add text mode
        text=T_A if show_labels else [str(i) for i in range(len(X_A))] if show_atom_indices else None,  # Add element labels
        textposition="top center",  # Position the text above the points
        marker=dict(
            size=[ATOMIC_SIZE[label]*20 for label in T_A], 
            color='red',
            symbol='circle'
        ),
        name=nameA
    ))

    # Add scatter points for structure B with different sizes for different atoms
    fig.add_trace(go.Scatter3d(
        x=X_B[:, 0],
        y=X_B[:, 1],
        z=X_B[:, 2],
        mode='markers+text' if show_labels or show_atom_indices else 'markers',  # Add text mode
        text=T_B if show_labels else [str(i) for i in range(len(X_B))] if show_atom_indices else None,  # Add element labels
        textposition="top center",  # Position the text above the points
        marker=dict(
            size=[ATOMIC_SIZE[label]*20 for label in T_B], 
            color='blue',
            symbol='circle'
        ),
        name=nameB
    ))

    # Add bonds for A
    for i in range(len(B_A)):
        for j in range(i+1, len(B_A)):
            if B_A[i,j] == 1:  # if there's a bond
                fig.add_trace(go.Scatter3d(
                    x=[X_A[i,0], X_A[j,0]],
                    y=[X_A[i,1], X_A[j,1]],
                    z=[X_A[i,2], X_A[j,2]],
                    mode='lines',
                    line=dict(
                        color='red',
                        width=2
                    ),
                    showlegend=False
                ))

    # Add bonds for B
    for i in range(len(B_B)):
        for j in range(i+1, len(B_B)):
            if B_B[i,j] == 1:  # if there's a bond
                fig.add_trace(go.Scatter3d(
                    x=[X_B[i,0], X_B[j,0]],
                    y=[X_B[i,1], X_B[j,1]],
                    z=[X_B[i,2], X_B[j,2]],
                    mode='lines',
                    line=dict(
                        color='blue',
                        width=2
                    ),
                    showlegend=False
                ))

    # Add matching lines between atoms
    if assignment is None:
        assignment = np.arange(len(X_A))
        print("The assignment is not provided. Assuming identity assignment.")
    for i in range(len(X_A)):
        j = assignment[i]
        fig.add_trace(go.Scatter3d(
            x=[X_A[i,0], X_B[j,0]],
            y=[X_A[i,1], X_B[j,1]],
            z=[X_A[i,2], X_B[j,2]],
            mode='lines',
            line=dict(
                color='green',
                width=3,
                dash='longdash'
                ),
            showlegend=False
            ))
        
    if True:
        #bonding_info_A = [B_A[i, j] for i in range(len(X_A)) for j in range(len(X_A)) if i < j]
        #bonding_info_B = [B_B[assignment[i], assignment[j]] for i in range(len(X_A)) for j in range(len(X_A)) if i < j]
        if not only_A_bonds:
            for i in range(len(X_A)):
                for j in range(len(X_A)):
                    if i < j and B_A[i,j] != B_B[assignment[i],assignment[j]]:
                        fig.add_trace(go.Scatter3d(
                            x=[X_A[i,0], X_A[j,0]],
                            y=[X_A[i,1], X_A[j,1]],
                            z=[X_A[i,2], X_A[j,2]],
                            mode='lines',
                            line=dict(
                                color='orange',
                                width=2
                            ),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter3d(
                            x=[X_B[assignment[i],0], X_B[assignment[j],0]],
                            y=[X_B[assignment[i],1], X_B[assignment[j],1]],
                            z=[X_B[assignment[i],2], X_B[assignment[j],2]],
                            mode='lines',
                            line=dict(
                                color='orange',
                                width=2
                            ),
                            showlegend=False
                        ))
        if only_A_bonds:
            for i in range(len(X_A)):
                for j in range(len(X_A)):
                    if i < j and B_A[i,j] == 1 and B_B[assignment[i],assignment[j]] == 0:
                        fig.add_trace(go.Scatter3d(
                            x=[X_A[i,0], X_A[j,0]],
                            y=[X_A[i,1], X_A[j,1]],
                            z=[X_A[i,2], X_A[j,2]],
                            mode='lines',
                            line=dict(
                                color='orange',
                                width=2
                            ),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter3d(
                            x=[X_B[assignment[i],0], X_B[assignment[j],0]],
                            y=[X_B[assignment[i],1], X_B[assignment[j],1]],
                            z=[X_B[assignment[i],2], X_B[assignment[j],2]],
                            mode='lines',
                            line=dict(
                                color='orange',
                                width=2
                            ),
                            showlegend=False
                        ))
    # Update layout
    fig.update_layout(
        title='{} {} Alignment'.format(nameA, nameB),
        scene=dict(
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis = dict(visible=False),
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
        ),
        showlegend=True,
    )

    # Save the figure as HTML
    if save:
        fig.write_html("{}_{}.html".format(nameA, nameB))

    config = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            #'filename': 'custom_image',
            'height': 500,
            'width': 700,
            'scale':2 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }
    # Show the figure
    fig.show(config=config)



