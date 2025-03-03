import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx.io.gmshio import model_to_mesh

from config import *

#-------------------------------------------------| mesh |-------------------------------------------------#
def generate_mesh(geometry, r, min_size, max_size, view=False):
    particleCenterList = geometry['centerList']
    
    gmsh.initialize()
    occ = gmsh.model.occ
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    # Set verbosity level to minimal (0 = no output)
    gmsh.option.setNumber("General.Verbosity", 0)

    if model_rank == 0:
        unit_cell = occ.add_box(0, 0, 0, 1, 1, 1)
        inclusions = [occ.addSphere(cntr[0], cntr[1], cntr[2], r) for cntr in particleCenterList]
        vol_dimTag = (gdim, unit_cell)
        out = occ.intersect(
            [vol_dimTag], [(gdim, incl) for incl in inclusions], removeObject=False
        )
        incl_dimTags = out[0]
        occ.synchronize()
        occ.cut([vol_dimTag], incl_dimTags, removeTool=False)
        occ.synchronize()

        # tag physical domains and facets
        gmsh.model.addPhysicalGroup(gdim, [vol_dimTag[1]], 1, name="Matrix")
        gmsh.model.addPhysicalGroup(
            gdim,
            [tag for _, tag in incl_dimTags],
            2,
            name="Inclusions",
        )

        # Get all faces of the box
        faces = occ.getEntities(dim=fdim)  # Get faces of the volume
        face_tags = [f[1] for f in faces]

        # Loop through the faces to identify their position by checking their vertices
        left_face = None
        right_face = None
        bottom_face = None
        top_face = None
        near_face = None
        far_face = None

        # Retrieve the vertices of the face using gmsh.model.getEntities
        # Loop through the faces and check the coordinates of their center of mass
        for face_tag in face_tags:
            coords = gmsh.model.occ.getCenterOfMass(fdim, face_tag)
            
            x, y, z = coords[:3]  # Get the 3D coordinates of the face center

            tol = 1e-6
            # Check for left, right, bottom, top, near, and far faces based on coordinates
            if np.abs(x - 0) < tol:  # Left face
                left_face = face_tag
            elif np.abs(x - 1) < tol:  # Right face
                right_face = face_tag
            if np.abs(y - 0) < tol:  # Bottom face
                bottom_face = face_tag
            elif np.abs(y - 1) < tol:  # Top face
                top_face = face_tag
            if np.abs(z - 0) < tol:  # Near face
                near_face = face_tag
            elif np.abs(z - 1) < tol:  # Far face
                far_face = face_tag

        # Assign the physical groups
        if left_face is not None:
            gmsh.model.addPhysicalGroup(fdim, [left_face], 3, name="Left")
        if right_face is not None:
            gmsh.model.addPhysicalGroup(fdim, [right_face], 4, name="Right")
        if bottom_face is not None:
            gmsh.model.addPhysicalGroup(fdim, [bottom_face], 5, name="Bottom")
        if top_face is not None:
            gmsh.model.addPhysicalGroup(fdim, [top_face], 6, name="Top")
        if near_face is not None:
            gmsh.model.addPhysicalGroup(fdim, [near_face], 7, name="Near")
        if far_face is not None:
            gmsh.model.addPhysicalGroup(fdim, [far_face], 8, name="Far")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)

        gmsh.model.mesh.generate(gdim)

    mesh, ct, ft = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
    if view:
        gmsh.fltk.run()
    gmsh.finalize()
    print("Mesh generated successfully")
    return mesh, ct, ft






