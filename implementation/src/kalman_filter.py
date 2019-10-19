import numpy as np


class KalmanFilter:
    """Implementation of ordinary Kalman filter."""

    def __init__(self,
                 initial_state: np.ndarray,
                 initial_covariance: np.ndarray,
                 process_covariance: np.ndarray,
                 measurement_covariance: np.ndarray,
                 transition_matrix: np.ndarray = None,
                 measurement_matrix: np.ndarray = None):
        self.x = initial_state
        self.P = initial_covariance

        self.Q = process_covariance
        self.R = measurement_covariance

        self.F = transition_matrix
        self.H = measurement_matrix

    def predict(self, transition_matrix: np.ndarray = None) -> np.ndarray:
        """
        Implements Kalman filter's predict step.
        :param transition_matrix: matrix describing system dynamics
        :return: the predicted state vector
        """

        if transition_matrix is None:
            if self.F is None:
                raise ValueError('Please provide transition matrix describing system dynamics.')
        else:
            self.F = transition_matrix

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray, measurement_matrix: np.ndarray = None):
        """
        Implements Kalman filter's update step using incoming measurement.
        :param z: the measurement vector
        :param measurement_matrix: matrix describing functional relationship between measurement
                                   and system state
        """
        if measurement_matrix is None:
            if self.H is None:
                raise ValueError('Please provide measurement matrix describing functional '
                                 'relationship between measurement and system state.')
        else:
            self.H = measurement_matrix

        # calculating Kalman gain
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # updating state
        self.x = self.x + np.dot(K, y)

        # updating state covariance
        I = np.eye(self.F.shape[1])
        self.P = np.add(
            np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T),
            np.dot(np.dot(K, self.R), K.T)
        )
