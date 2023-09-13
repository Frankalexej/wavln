import numpy as np

class DirectionalConsistency: 
    @staticmethod
    def analyze(trajectory): 
        if trajectory.shape[0] <= 2: 
            return 0.
        
        # Compute direction vectors between consecutive points
        direction_vectors = np.diff(trajectory, axis=0)

        # Calculate angles between consecutive direction vectors
        angles = []
        for i in range(1, len(direction_vectors)):
            v1 = direction_vectors[i - 1]
            v2 = direction_vectors[i]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angles.append(angle)

        # Calculate variance of the angles
        variance = np.var(angles)

        return variance