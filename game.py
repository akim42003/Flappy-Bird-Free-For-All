import pygame
import random

# Game constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
PIPE_GAP = 180
PIPE_WIDTH = 55
PIPE_SPEED = 6
BIRD_SIZE = 28
GRAVITY = 0.75
FLAP_STRENGTH = -8
FPS = 30


class FlappyBirdGame:
    def __init__(self, headless=False):
        pygame.init()
        if not headless:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))  # Off-screen rendering
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()

        self.bird = None
        self.pipes = None
        self.score = 0
        self.reset()

    def reset(self):
        """Reset the game state and return the initial state."""
        # Reset bird and pipes
        self.bird = {"x": 100, "y": SCREEN_HEIGHT // 2, "velocity": 0}
        self.pipes = [
            {"x": SCREEN_WIDTH + 100, "gap_y": random.randint(100, SCREEN_HEIGHT - PIPE_GAP - 100)}
        ]
        self.score = 0

        # Render the initial state
        self.render()

        return self._get_state()

    def step(self, action):
        """Perform an action and return (next_state, reward, done)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        reward = 0
        done = False

        # Bird action
        if action == 1:  # Flap
            self.bird["velocity"] = FLAP_STRENGTH

        # Bird physics
        self.bird["velocity"] += GRAVITY
        self.bird["y"] += self.bird["velocity"]

        # Pipe movement
        for pipe in self.pipes:
            pipe["x"] -= PIPE_SPEED

        # Add new pipes and remove off-screen ones
        if self.pipes[0]["x"] < -PIPE_WIDTH:
            self.pipes.pop(0)
            self.pipes.append({
                "x": SCREEN_WIDTH,
                "gap_y": random.randint(100, SCREEN_HEIGHT - PIPE_GAP - 100)
            })
            reward += 25  # Reward for passing a pipe
            self.score += 1

        # Check collisions
        if self._check_collision():
            done = True
            reward -= 100  # Penalty for collision

        # Check if bird is out of bounds
        if self.bird["y"] < 0 or self.bird["y"] > SCREEN_HEIGHT:
            done = True
            reward -= 150  # Penalty for going out of bounds

        # Render the game
        self.render()

        # Get the next state
        next_state = self._get_state()
        return next_state, reward, done

    def render(self):
        """Render the game objects on the screen."""
        # Clear the screen
        self.screen.fill((0, 0, 0))  # Black background

        # Draw the pipes
        for pipe in self.pipes:
            # Top pipe
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe["x"], 0, PIPE_WIDTH, pipe["gap_y"]))
            # Bottom pipe
            pygame.draw.rect(self.screen, (0, 255, 0),
                             (pipe["x"], pipe["gap_y"] + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT - pipe["gap_y"] - PIPE_GAP))

        # Draw the bird
        pygame.draw.rect(self.screen, (255, 0, 0), (self.bird["x"], self.bird["y"], BIRD_SIZE, BIRD_SIZE))

        # Display the score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Update the display
        pygame.display.flip()

    def _get_state(self):
        """Return the current game state as [bird_y, velocity, pipe_x_distance, pipe_y_distance]."""
        pipe = self.pipes[0]
        return [
            self.bird["y"],
            self.bird["velocity"],
            pipe["x"] - self.bird["x"],
            pipe["gap_y"] - self.bird["y"]
        ]

    def _check_collision(self):
        """Check if the bird collides with a pipe or the screen boundaries."""
        pipe = self.pipes[0]
        bird_rect = pygame.Rect(self.bird["x"], self.bird["y"], BIRD_SIZE, BIRD_SIZE)
        top_pipe_rect = pygame.Rect(pipe["x"], 0, PIPE_WIDTH, pipe["gap_y"])
        bottom_pipe_rect = pygame.Rect(pipe["x"], pipe["gap_y"] + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT - pipe["gap_y"] - PIPE_GAP)
        return bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect)


if __name__ == "__main__":
    game = FlappyBirdGame()
    running = True
    action = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1

        state, reward, done = game.step(action)
        action = 0  # Reset action after each frame

        if done:
            print("Game Over! Resetting...")
            game.reset()

        game.clock.tick(FPS)

    pygame.quit()
