import random
import math
import numpy as np

#-------------------------------------------------| geometry |---------------------------------------------#
def generate_random_geometry(volume_fraction, num_particles, r, delDistanceMultiplier, delDistanceBorder):
    L=1
    
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    def is_overlapping(particle, particles):
        for p in particles:
            if distance(p[0], particle[0]) < (p[1] + particle[1]) * delDistanceMultiplier:
                return True
        return False
    
    def intersects_corner(particle):
        x, y, z = particle[0]
        r = particle[1]
        corners = [(0, 0, 0), (0, 0, L), (0, L, 0), (0, L, L), (L, 0, 0), (L, 0, L), (L, L, 0), (L, L, L)]
        for corner in corners:
            if distance((x, y, z), corner) <= r:
                return True
        return False

    def intersects_edge(particle):
        x, y, z = particle[0]
        r = particle[1]
        edges = [
            ((0, 0, 0), (L, 0, 0)), ((0, 0, 0), (0, L, 0)), ((0, 0, 0), (0, 0, L)),
            ((0, L, 0), (L, L, 0)), ((0, L, 0), (0, L, L)), ((L, 0, 0), (L, L, 0)),
            ((L, 0, 0), (L, 0, L)), ((0, 0, L), (L, 0, L)), ((0, 0, L), (0, L, L)),
            ((0, L, L), (L, L, L)), ((L, 0, L), (L, L, L)), ((L, L, 0), (L, L, L))
        ]
        for edge in edges:
            for i in range(3):
                if edge[0][i]-edge[1][i]!=0:
                    axes=i
            midpoint = tuple((c1 + c2) / 2 for c1, c2 in zip(edge[0], edge[1]))
            midpoint[i]==particle[0][i]
            theDist = distance((x,y,z),midpoint)
            if theDist <= r:
                return True
        return False
    
    def intersects_face(particle):
        x, y, z = particle[0]
        r = particle[1]
        faces = [
            (x, 0, z), (x, L, z), (x, y, 0), (x, y, L), (0, y, z), (L, y, z)
        ]
        face_intersections = []
        for i, face in enumerate(faces):
            if distance((x, y, z), face) <= r:
                face_intersections.append(i)
        return len(face_intersections) > 0, face_intersections
    
    def intersects_allBoundary(particle):
        corner = intersects_corner(particle)
        if corner:
            return True, ['corner']
        
        edge = intersects_edge(particle)
        if edge:
            return True, ['edge']
        
        face = intersects_face(particle)
        if face[0]:
            return True, ['face',face[1]]
        
        return False, []
    
    def closest_corner(center):
        # Get the closest corner to the given center
        corner_coords = [
            (0, 0, 0), (0, 0, L), (0, L, 0), (0, L, L),
            (L, 0, 0), (L, 0, L), (L, L, 0), (L, L, L)
        ]
        closest_corner = min(corner_coords, key=lambda c: distance(center, c))
        return closest_corner
    
    def reproduce_in_corners(particle):
        # Get the center and radius of the given particle
        center, radius = particle
        
        # Find the closest corner to the particle's center
        reference_corner = closest_corner(center)
        
        # Compute the displacement vector from the closest corner to the origin
        origin_displacement = (-reference_corner[0], -reference_corner[1], -reference_corner[2])
        
        # Get the coordinates of all eight corners of the unit cell
        corners = [
            (0, 0, 0), (0, 0, L), (0, L, 0), (0, L, L),
            (L, 0, 0), (L, 0, L), (L, L, 0), (L, L, L)
        ]
        
        # Reproduce the particle at each corner considering the closest corner
        reproduced_particles = [((x + center[0] + origin_displacement[0], 
                                y + center[1] + origin_displacement[1], 
                                z + center[2] + origin_displacement[2]), radius) for x, y, z in corners]
        return reproduced_particles
    
    def closest_edge(center):
        # Define all edges as pairs of points
        edges = [
            ((0, 0, 0), (L, 0, 0)), ((0, 0, 0), (0, L, 0)), ((0, 0, 0), (0, 0, L)),
            ((0, L, 0), (L, L, 0)), ((0, L, 0), (0, L, L)), ((L, 0, 0), (L, L, 0)),
            ((L, 0, 0), (L, 0, L)), ((0, 0, L), (L, 0, L)), ((0, 0, L), (0, L, L)),
            ((0, L, L), (L, L, L)), ((L, 0, L), (L, L, L)), ((L, L, 0), (L, L, L))
        ]
        
        # Find the closest edge to the center
        min_distance = float('inf')
        closest_edge = None
        for edge in edges:
            midpoint = tuple((c1 + c2) / 2 for c1, c2 in zip(edge[0], edge[1]))
            d = distance(center, midpoint)
            if d < min_distance:
                min_distance = d
                closest_edge = edge
        return closest_edge
    
    def reproduce_in_edges(particle):
        # Get the center and radius of the given particle
        center, radius = particle
        
        # Find the closest edge to the particle's center
        reference_edge = closest_edge(center)
        
        # Determine the direction of the reference edge
        direction = tuple(b - a for a, b in zip(*reference_edge))
        
        # Define all edges as pairs of points
        edges = [
            ((0, 0, 0), (L, 0, 0)), ((0, 0, 0), (0, L, 0)), ((0, 0, 0), (0, 0, L)),
            ((0, L, 0), (L, L, 0)), ((0, L, 0), (0, L, L)), ((L, 0, 0), (L, L, 0)),
            ((L, 0, 0), (L, 0, L)), ((0, 0, L), (L, 0, L)), ((0, 0, L), (0, L, L)),
            ((0, L, L), (L, L, L)), ((L, 0, L), (L, L, L)), ((L, L, 0), (L, L, L))
        ]
        
        # Filter the edges to find those parallel to the reference edge
        parallel_edges = [edge for edge in edges if tuple(b - a for a, b in zip(*edge)) == direction]
        
        # Compute the displacement vector from the reference edge's midpoint to the origin
        midpoint = tuple((c1 + c2) / 2 for c1, c2 in zip(reference_edge[0], reference_edge[1]))
        origin_displacement = (-midpoint[0], -midpoint[1], -midpoint[2])
        
        # Reproduce the particle at the midpoint of each parallel edge considering the reference edge's displacement
        reproduced_particles = [((0.5 * (edge[0][0] + edge[1][0]) + center[0] + origin_displacement[0],
                                0.5 * (edge[0][1] + edge[1][1]) + center[1] + origin_displacement[1],
                                0.5 * (edge[0][2] + edge[1][2]) + center[2] + origin_displacement[2]), radius) for edge in parallel_edges]
        return reproduced_particles
    
    def closest_face(center):
        # Define the center of each face of the unit cell
        faces = [
            (L / 2, 0, L / 2), (L / 2, L, L / 2), (L / 2, L / 2, 0),
            (L / 2, L / 2, L), (0, L / 2, L / 2), (L, L / 2, L / 2)
        ]
        
        # Find the closest face center to the particle's center
        closest_face_center = min(faces, key=lambda face: distance(center, face))
        return closest_face_center

    def reproduce_in_faces(particle):
        # Get the center and radius of the given particle
        center, radius = particle
        
        # Find the closest face center to the particle's center
        reference_face_center = closest_face(center)
        
        # Compute the displacement vector from the closest face center to the origin
        origin_displacement = (-reference_face_center[0], -reference_face_center[1], -reference_face_center[2])
        
        # Define the center of each face of the unit cell
        faces = [
            (L / 2, 0, L / 2), (L / 2, L, L / 2), (L / 2, L / 2, 0),
            (L / 2, L / 2, L), (0, L / 2, L / 2), (L, L / 2, L / 2)
        ]
        
        # Determine the normal of the reference face
        normal = tuple(fc - oc for fc, oc in zip(reference_face_center, (L/2, L/2, L/2)))
        for element in normal:
            if element != 0:
                axis = normal.index(element)
                #print(f"axis found: {axis}")
        
        parallel_faces_normal = [0,0,0]
        parallel_faces_normal[axis]-=normal[axis]
        parallel_faces_normal = tuple(parallel_faces_normal)
        #print(f"parallel_faces_normal: {parallel_faces_normal}")

        # Filter the faces to find those parallel to the reference face
        parallel_faces = [face for face in faces if tuple(fc - oc for fc, oc in zip(face, (L/2, L/2, L/2))) == parallel_faces_normal]
        
        # Reproduce the particle at the center of each parallel face considering the reference face center's displacement
        reproduced_particles = [((face[0] + center[0] + origin_displacement[0],
                                face[1] + center[1] + origin_displacement[1],
                                face[2] + center[2] + origin_displacement[2]), radius) for face in parallel_faces]
        return reproduced_particles
    
    def reproduce_in_dual_faces(particle, intersected_faces):
        center, radius = particle
        reproduced_particles = []
        
        # Check which faces are intersected and reproduce accordingly
        if 0 in intersected_faces and 1 in intersected_faces:  # Bottom and Top (y = 0 and y = L)
            # Place at the corners of y = 0 and y = L
            reproduced_particles.append(((center[0], 0, center[2]), radius))  # on y = 0
            reproduced_particles.append(((center[0], L, center[2]), radius))  # on y = L
            
        if 2 in intersected_faces and 3 in intersected_faces:  # Front and Back (z = 0 and z = L)
            reproduced_particles.append(((center[0], center[1], 0), radius))  # on z = 0
            reproduced_particles.append(((center[0], center[1], L), radius))  # on z = L
        
        if 4 in intersected_faces and 5 in intersected_faces:  # Left and Right (x = 0 and x = L)
            reproduced_particles.append(((0, center[1], center[2]), radius))  # on x = 0
            reproduced_particles.append(((L, center[1], center[2]), radius))  # on x = L
        
        return reproduced_particles

    def sphere_volume_in_face(particle):
        center,radius=particle
        def sphere_cap_volume(h, r):
            return (math.pi * h**2 * (3*r - h)) / 3
        
        x, y, z = center
        full_volume = (4/3) * math.pi * radius**3

        # Check intersection with each face of the cube and subtract the corresponding cap volume if necessary
        intersection_volume = full_volume

        # Faces at x = 0 and x = L
        if x - radius < 0:
            cap_height = radius - x
            intersection_volume -= sphere_cap_volume(cap_height, radius)
        if x + radius > L:
            cap_height = radius - (L - x)
            intersection_volume -= sphere_cap_volume(cap_height, radius)

        # Faces at y = 0 and y = L
        if y - radius < 0:
            cap_height = radius - y
            intersection_volume -= sphere_cap_volume(cap_height, radius)
        if y + radius > L:
            cap_height = radius - (L - y)
            intersection_volume -= sphere_cap_volume(cap_height, radius)

        # Faces at z = 0 and z = L
        if z - radius < 0:
            cap_height = radius - z
            intersection_volume -= sphere_cap_volume(cap_height, radius)
        if z + radius > L:
            cap_height = radius - (L - z)
            intersection_volume -= sphere_cap_volume(cap_height, radius)

        return max(intersection_volume, 0)
    
    def sphere_volume_in_corner(particle):
        center,radius=particle
        x, y, z = center
        if x - radius < 0 and y - radius < 0 and z - radius < 0:
            return (1/8) * (4/3) * math.pi * radius**3
        if x + radius > L and y - radius < 0 and z - radius < 0:
            return (1/8) * (4/3) * math.pi * radius**3
        if x - radius < 0 and y + radius > L and z - radius < 0:
            return (1/8) * (4/3) * math.pi * radius**3
        if x - radius < 0 and y - radius < 0 and z + radius > L:
            return (1/8) * (4/3) * math.pi * radius**3
        if x + radius > L and y + radius > L and z - radius < 0:
            return (1/8) * (4/3) * math.pi * radius**3
        if x + radius > L and y - radius < 0 and z + radius > L:
            return (1/8) * (4/3) * math.pi * radius**3
        if x - radius < 0 and y + radius > L and z + radius > L:
            return (1/8) * (4/3) * math.pi * radius**3
        if x + radius > L and y + radius > L and z + radius > L:
            return (1/8) * (4/3) * math.pi * radius**3
        return 0
    
    def sphere_volume_in_edge(particle):
        center,radius=particle
        def sphere_cap_volume(h, r):
            return (math.pi * h**2 * (3*r - h)) / 3
        
        x, y, z = center
        full_volume = (4/3) * math.pi * radius**3

        # Check intersection with each face of the cube and subtract the corresponding cap volume if necessary
        intersection_volume = full_volume

        # Edges at x = 0, L; y = 0, L; z = 0, L
        if (x - radius < 0 and y - radius < 0) or (x - radius < 0 and z - radius < 0) or (y - radius < 0 and z - radius < 0):
            cap_height = radius - min(x, y, z)
            intersection_volume -= (1/4) * sphere_cap_volume(cap_height, radius)
        if (x + radius > L and y + radius > L) or (x + radius > L and z + radius > L) or (y + radius > L and z + radius > L):
            cap_height = radius - (L - max(x, y, z))
            intersection_volume -= (1/4) * sphere_cap_volume(cap_height, radius)
        if (x - radius < 0 and y + radius > L) or (x - radius < 0 and z + radius > L) or (y - radius < 0 and z + radius > L):
            cap_height = radius - min(x, L - y, z)
            intersection_volume -= (1/4) * sphere_cap_volume(cap_height, radius)
        if (x + radius > L and y - radius < 0) or (x + radius > L and z - radius < 0) or (y + radius > L and z - radius < 0):
            cap_height = radius - min(L - x, y, z)
            intersection_volume -= (1/4) * sphere_cap_volume(cap_height, radius)

        return max(intersection_volume, 0)
    
    def generate_particles(volume_fraction, num_particles, max_iterations=1e3):
        particles = []
        num = 0
        totalVolume = 0
        iteration = 0  # Track the number of iterations
        while num < num_particles and iteration < max_iterations:
            radius = r
            center = (random.uniform(0 + delDistanceBorder, 1 - delDistanceBorder), random.uniform(0 + delDistanceBorder, 1 - delDistanceBorder), random.uniform(0 + delDistanceBorder, 1 - delDistanceBorder))
            particle = (center, radius)
            if not is_overlapping(particle, particles): # and particle[1]<=rMax and particle[1]>=rMin:
                isBoundary, modes = intersects_allBoundary(particle)
                if isBoundary:
                    if modes[0] == "corner":
                        reproduced = reproduce_in_corners(particle)
                        check = 0
                        reproVolume = 0
                        newTotalNumParticles = 0
                        for part in reproduced:
                            if not is_overlapping(part, particles):
                                check += 1
                                reproVolume+=sphere_volume_in_corner(part)
                                newTotalNumParticles+=1
                        if newTotalNumParticles>num_particles:
                            continue
                        if check == len(reproduced) and len(reproduced)<=num_particles-num and totalVolume + reproVolume + sphere_volume_in_edge(particle)<=L**3*volume_fraction:
                            particles.append(particle)
                            num += 1
                            #Log(f"og corner part: {particle}")
                            for part in reproduced:
                                particles.append(part)
                                num += 1
                                #Log(f"reproduced corner part: {part}")
                    elif modes[0] == "edge":
                        reproduced = reproduce_in_edges(particle)
                        check = 0
                        reproVolume = 0
                        for part in reproduced:
                            if not is_overlapping(part, particles):
                                check += 1
                                reproVolume+=sphere_volume_in_edge(part)
                        if check == len(reproduced) and len(reproduced)<=num_particles-num and totalVolume + reproVolume + sphere_volume_in_edge(particle)<=L**3*volume_fraction:
                            particles.append(particle)
                            num += 1
                            #Log(f"og edge part: {particle}")
                            for part in reproduced:
                                particles.append(part)
                                num += 1
                                #Log(f"reproduced edge part: {part}")
                        
                    elif modes[0] == "face":
                        numOfIntersections = modes[1]
                        if len(numOfIntersections)==1:
                            reproduced = reproduce_in_faces(particle)
                            check = 0
                            reproVolume = 0
                            newTotalNumParticles = num
                            for part in reproduced:
                                if not is_overlapping(part, particles):
                                    check += 1
                                    reproVolume+=sphere_volume_in_face(part)
                                    newTotalNumParticles+=1
                            if newTotalNumParticles+1<num_particles:
                                if check == len(reproduced) and len(reproduced)<=num_particles-num and totalVolume + reproVolume + sphere_volume_in_face(particle)<=L**3*volume_fraction:
                                    particles.append(particle)
                                    num += 1
                                    #Log(f"og face part: {particle}")
                                    for part in reproduced:
                                        particles.append(part)
                                        num += 1
                                        #Log(f"reproduced face part: {part}")
                        elif len(numOfIntersections)==2:
                            reproduced = reproduce_in_dual_faces(particle, modes[1])
                            check = 0
                            reproVolume = 0
                            newTotalNumParticles = num
                            for part in reproduced:
                                if not is_overlapping(part, particles):
                                    check += 1
                                    reproVolume+=sphere_volume_in_face(part)
                                    newTotalNumParticles+=1
                            if newTotalNumParticles+1>num_particles:
                                if check == len(reproduced) and len(reproduced)<=num_particles-num and totalVolume + reproVolume + sphere_volume_in_face(particle)<=L**3*volume_fraction:
                                    particles.append(particle)
                                    num += 1
                                    #Log(f"og face part: {particle}")
                                    for part in reproduced:
                                        particles.append(part)
                                        num += 1
                                        #Log(f"reproduced face part: {part}")
                elif not isBoundary:
                    if totalVolume+4/3*np.pi*r**3<=volume_fraction:
                        totalVolume+=4/3*np.pi*r**3
                        particles.append(particle)
                        num += 1
                        #Log(f"direct particle: {particle}")
            
            iteration += 1  # Increment the iteration count
            
        # If the loop exits and hasn't completed all particles, show a warning
        if iteration >= max_iterations:
            print(f"Warning: Reached maximum iterations ({max_iterations}) before placing all particles.")

        return particles

    success = False
    retries = 0
    while not success and retries < 10:
        try:
            particles = generate_particles(volume_fraction, num_particles, max_iterations=1e4)
            success = True
        except Exception as e:
            retries += 1
            print(f"An error occurred while generating geometry. Retrying... (Attempt {retries}/10)")

    if not success:
        print("Failed to generate geometry after multiple attempts.")
        return None

    centerCoordinatesList = [center for center, radius in particles]
    radiusList = [radius for center, radius in particles]
    print(f"Geometry generated successfully")
    return {"centerList": centerCoordinatesList, "radiusList": radiusList}
