import torch
import pygame
from game import FlappyBirdGame  


def watch_agent_play(agent, episodes=1):
    """Train the agent"""
    game = FlappyBirdGame(headless=False)  
    for episode in range(episodes):
        state = game.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            state, reward, done = game.step(action)
            total_reward += reward
            game.clock.tick(30)  
        
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")


def play_manually():
    """Allow manual interaction with the game."""
    game = FlappyBirdGame(headless=False)
    running = True

    while running:
        action = 0  # Default action is "do nothing"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1  # Flap action

        state, reward, done = game.step(action)
        
        if done:
            print("Game Over!")
            running = False

        game.clock.tick(30)  

    pygame.quit()
    print("Exited manual play.")


def main():
    """Interactive menu for selecting inference or manual play."""
    print("=== Flappy Bird Interactive Menu ===")
    print("1. Train agent ")
    print("2. Play the game manually")
    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        try:
            from dqnn import DQLAgent 
            agent = DQLAgent(input_dim=4, output_dim=2, device="cpu")  
            agent.q_network.load_state_dict(torch.load("flappy_dql_model.pth"))
            agent.q_network.eval()
            print("Model loaded successfully! Watching the agent play...")
            watch_agent_play(agent, episodes=5)
        except FileNotFoundError:
            print("Error: Trained model file 'flappy_dql_model.pth' not found. Please train the model first.")
    elif choice == "2":
        print("Playing manually... Press SPACE to flap!")
        play_manually()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
