import numpy as np
import cv2


class Boids:
    """Boids Algorithm (Fish ver.)"""

    def __init__(self, fish_num, field_height=720, field_width=960, fish_len=13, ave_speed=1.5,
                 stimulated=False, draw_fish_circle=False, dt=0.5):
        self.dt = dt

        # Field parameter
        self.height = field_height
        self.width = field_width
        self.draw_fish_circle = draw_fish_circle

        # Fish parameter
        self.fish_num = fish_num
        self.fish_len = fish_len
        # Distance of various areas of fish
        self.repulsive_radius = 1 * fish_len
        self.parallel_radius = 12 * fish_len
        self.attraction_radius = 15 * fish_len
        # Position of fish
        fish_x = np.random.rand(fish_num, 1) * field_width
        fish_y = np.random.rand(fish_num, 1) * field_height
        self.fish_coord = np.concatenate((fish_x, fish_y), axis=1)
        # Fish velocity: (r, θ) = (fish_speed, fish_direction)
        self.ave_speed = ave_speed * fish_len  # Average speed of fish
        self.fish_speed = np.random.normal(self.ave_speed, size=fish_num)
        self.fish_direction = np.random.rand(fish_num) * (2 * np.pi)
        # Distance and angle between fish
        self.fish_distance = np.full((fish_num, fish_num), np.inf)
        self.fish_angle = np.zeros((fish_num, fish_num))
        self.beta = np.zeros((fish_num, fish_num))

        # Stimulus parameter
        self.stimulated = stimulated
        self.stimulus_radius = 15 * fish_len
        self.stimulus_coord = np.zeros(2)
        # Distance and angle between the fish and the stimulus
        self.stimulus_distance = np.full(fish_num, np.inf)
        self.stimulus_angle = np.zeros(fish_num)

        # Fish polygon used to draw fish
        self.fish_polygon = np.array([(0, 0), (-1, 0.5), (-3, 0), (-1, -0.5)])
        self.fish_polygon *= fish_len / 3
        # Variables used to draw a stimulus circle
        self.count = 0

    @staticmethod
    def gaussian(distance, scale=1.0):
        """Simplified Gaussian function"""
        return np.exp(- distance / (2 * scale ** 2))

    def move(self, fluctuation_max=np.pi / 20, inertia=0.5):
        # Calculate relative coordinates
        fish_table = np.resize(self.fish_coord, (self.fish_num, self.fish_num, 2))
        relative_coord = fish_table - fish_table.transpose((1, 0, 2))
        # Calculate the distance between fish i, j
        self.fish_distance = np.linalg.norm(relative_coord, axis=2) + np.eye(self.fish_num) * self.attraction_radius
        # Calculate the angle [rad] between fish i and fish j (do not consider the connection outside the field)
        self.fish_angle = - np.arctan2(relative_coord[:, :, 1], relative_coord[:, :, 0])

        if self.stimulated:
            fish_table = np.resize(self.fish_coord, (self.fish_num, 2))
            stimulus_table = np.resize(self.stimulus_coord, (self.fish_num, 2))
            relative_coord = fish_table - stimulus_table

            # Calculate the distance and angle between the fish and the stimulus
            self.stimulus_distance = np.linalg.norm(relative_coord, axis=1)
            self.stimulus_angle = - np.arctan2(relative_coord[:, 1], relative_coord[:, 0])

            # The fish turns to avoid stimulus
            normal_distance = self.stimulus_distance / self.stimulus_radius  # Normalized distance
            self.fish_direction = np.where(self.stimulus_distance < self.stimulus_radius,
                                           -self.stimulus_angle * self.gaussian(normal_distance, inertia),
                                           self.fish_direction)

        # Create an array seq1 that represents the probabilities needed to randomly select fish
        seq1 = self.gaussian(self.fish_distance, 2.3)
        seq1 /= np.sum(seq1, axis=0)
        # Create the random variable seq2 needed to update the speed
        seq2 = - self.gaussian(self.fish_distance, 1.0) + 1
        # Create the parameter beta required to update the direction
        fish_table = np.resize(self.fish_direction, (self.fish_num, self.fish_num))
        fish_table -= fish_table.T
        # out of attractive area
        self.beta = (np.random.random(self.beta.shape) - 0.5) * np.pi / 6
        # in attractive area
        self.beta = np.where(self.fish_distance <= self.attraction_radius, self.fish_angle, self.beta)
        # in parallel area
        self.beta = np.where(self.fish_distance <= self.parallel_radius, fish_table, self.beta)
        # in repulsive area
        self.beta = np.where(self.fish_distance <= self.repulsive_radius, -fish_table, self.beta)

        # Randomly generate fluctuations according to a normal distribution
        fluctuation = np.sqrt(2) * fluctuation_max * np.random.normal(size=self.fish_num)

        for i in range(self.fish_num):
            # Select one fish j (the closer it is to fish i, the easier it is to select)
            j = np.random.choice(self.fish_num, p=seq1[:, i])
            # Update speed and direction
            if j != i:
                self.fish_speed[i] = seq2[i][j] * self.ave_speed
                self.fish_direction[i] += (self.beta[i, j] + fluctuation[i]) * self.dt

        # Normalize the angle to the range -π to π
        self.fish_direction = np.arctan2(np.sin(self.fish_direction), np.cos(self.fish_direction))
        self.fish_angle = np.arctan2(np.sin(self.fish_angle), np.cos(self.fish_angle))

        # Convert polar speed to Cartesian speed
        rectangular_speed = np.array([self.fish_speed * np.cos(self.fish_direction),
                                      self.fish_speed * np.sin(self.fish_direction)])

        # Update coordinates (position)
        self.fish_coord += rectangular_speed.T * self.dt

        # When the fish go out of the field, it comes out from the other side
        self.fish_coord[:, 0] = np.mod(self.fish_coord[:, 0], self.width)
        self.fish_coord[:, 1] = np.mod(self.fish_coord[:, 1], self.height)

    def draw(self, circle_speed=0.05):
        # Draw the field
        field = np.full((self.height, self.width, 3), (57, 30, 19), np.uint8)

        # Draw circle of sight of the fish
        if self.draw_fish_circle:
            for i in range(self.fish_num):
                center = tuple(self.fish_coord[i].astype(int))
                cv2.circle(field, center, self.attraction_radius, (51, 51, 13))

        # Draw the fish
        for i in range(self.fish_num):
            rotation_matrix = np.array([[np.cos(self.fish_direction[i]), np.sin(self.fish_direction[i])],
                                        [-np.sin(self.fish_direction[i]), np.cos(self.fish_direction[i])]])
            fish = np.dot(self.fish_polygon, rotation_matrix) + self.fish_coord[i]

            if 0 <= i % 10 <= 4:
                fish_color = (255, 175, 140)
            elif 4 <= i % 10 <= 7:
                fish_color = (255, 142, 169)
            else:
                fish_color = (255, 255, 127)
            cv2.fillPoly(field, [fish.astype(int)], fish_color)

        # Draw the stimulus
        if self.stimulated:
            center = tuple(self.stimulus_coord.astype(int))
            cv2.circle(field, center, int(np.sin(self.count) * self.attraction_radius), (175, 85, 106), thickness=2)
            self.count = (self.count + circle_speed) % (np.pi // 2)

        return field


if __name__ == "__main__":
    boids = Boids(fish_num=300, stimulated=True)

    def move_stimulus(event, x, y, flags, param):
        boids.stimulus_coord = np.array((x, y))

    cv2.namedWindow('boids')
    cv2.setMouseCallback('boids', move_stimulus)
    while True:
        boids.move()
        cv2.imshow("boids", boids.draw(circle_speed=0.05))
        cv2.waitKey(delay=1)
